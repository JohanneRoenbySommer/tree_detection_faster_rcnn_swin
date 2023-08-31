import os
import pandas as pd
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
from shapely import wkt
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.cluster import AgglomerativeClustering

def load_data(filename):
    # Load data
    cols = ['id', 
            'kronediameter', 
            'traeart', 
            'dansk_navn', 
            'slaegt', 
            'slaegtsnavn', 
            'planteaar', 
            'reg_dato', 
            'opdateret_dato',
            'kategori', 
            'element', 
            'ny_dm_element', 
            'ny_dm_under_element',
            'busk_trae', 
            'bydelsnavn',
            'wkb_geometry']
    df = pd.read_csv(filename, usecols=cols)

    # Convert to GeoDataFrame
    df.rename(columns={'wkb_geometry': 'geometry'}, inplace=True)
    cols = cols[:-1] + ['geometry']
    df['geometry'] = df['geometry'].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, crs='epsg:4326', geometry='geometry')

    # Change dtype of year column
    gdf['planteaar'].fillna(0, inplace=True)
    gdf['planteaar'] = gdf['planteaar'].astype('int')

    # Replace empty columns with nan
    str_cols = [col for col in cols if col not in ['id', 'reg_dato', 'planteaar', 'geometry']]
    gdf[str_cols] = gdf[str_cols].apply(lambda x: x.str.strip()).replace('Ikke registreret', np.nan).replace('Ikke defineret', np.nan).replace('', np.nan)

    # Add coordinates as seperate columns
    gdf['lat'] = gdf['geometry'].y
    gdf['lng'] = gdf['geometry'].x
    
    return gdf

def remove_duplicates(gdf):
    # Cluster together trees that are within 0.5 meter distance from each other

    clustering = AgglomerativeClustering(n_clusters=None, 
                                         linkage='single',
                                         affinity='euclidean',
                                         distance_threshold=0.5)
    clustering.fit(np.vstack(gdf.to_crs('EPSG:25832').apply(lambda row: [row['geometry'].x, row['geometry'].y], axis=1).values))
    gdf['cluster'] = clustering.labels_

    # Only use the newest planted and registered tree when there are multiple trees located close to each other
    gdf_wo_dup = gdf.groupby('cluster', group_keys=False).apply(lambda x: x.sort_values(by=['planteaar', 'reg_dato'], ascending=False).iloc[:1])
    del gdf_wo_dup['cluster']

    return gdf_wo_dup

def remove_bushes(gdf_wo_dup):
    return gdf_wo_dup[gdf_wo_dup['busk_trae'] == 'Træ']

def remove_irrelevant(gdf_wo_dup):
    # Remove park trees
    gdf_wo_dup = gdf_wo_dup[(gdf_wo_dup['kategori'] != 'parktræ') & (gdf_wo_dup['ny_dm_element'] != 'Park- og naturtræer') & (~gdf_wo_dup['ny_dm_under_element'].isin(['Parktræ', 'Naturtræ']))]

    # Remove private trees
    gdf_wo_dup = gdf_wo_dup[(gdf_wo_dup['kategori'] != 'privat træ') & (gdf_wo_dup['ny_dm_element'] != 'Private træer')]

    # Remove trees in stands
    gdf_wo_dup = gdf_wo_dup[gdf_wo_dup['ny_dm_element'] != 'Træer i bevoksninger']

    return gdf_wo_dup

def scrape_info(species_name, species_name_map):
    if ' sp.' in species_name:
        col = False
    else:
        # Replace species name by the one used at vdberk.com
        if species_name in species_name_map:
            species_name = species_name_map[species_name]

        # Clean species name
        species_name_split = [w for w in species_name.split() if w not in ['hybr.', 'x', "'"]]
        species_name_split = [w.strip("'").strip('"').strip('.').lower().replace("'", '-').replace('ø', 'oe') for w in species_name_split]
    
        # Get species info from vdberk.com
        species_name_new = '-'.join(species_name_split)
        url = f'https://www.vdberk.com/trees/{species_name_new}/'
        resp = requests.get(url)
        soup = BeautifulSoup(resp.content, 'html.parser')
        col = soup.find(lambda tag: tag.name == 'div' and tag.get('class') == ['col-12'])
    
    if not col:
        return None
    
    for i, div in enumerate(col):
        # Get height and growth information
        if i == 1:
            height_splitted = div.text.split(',')
            # Get max height
            max_height_splitted = height_splitted[0].split()
            # If there's no space between height and 'm'
            if len(max_height_splitted[-1]) > 1:
                max_height_splitted = max_height_splitted[:-1] + [max_height_splitted[-1].strip('.')[:-1], max_height_splitted[-1].strip('.')[-1:]]
            max_height = max_height_splitted[-2].replace('(', '').replace(')', '').split('-')[-1]
            height_growth_info = [int(max_height)]
        elif i > 1:
            break
    
    return {'max_height': height_growth_info}

def get_species_info(gdf_wo_dup, species_name_map, species_manual_height):
    # Get max height of each species
    size_mappings = dict()
    species_names = gdf_wo_dup.groupby('traeart').size().sort_values(ascending=False).index
    for species_name in species_names:
        species_info = scrape_info(species_name, species_name_map)
        if species_info:
            size_mappings[species_name] = species_info

    # Manually add height for species not present at vdberk.com
    size_mappings.update({key: {'max_height': [val]} for key, val in species_manual_height.items()})#, 'crown': ['']} for key, val in species_manual_height.items()})

    # Add info on species to trees
    gdf_wo_dup['max_height'] = gdf_wo_dup['traeart'].apply(lambda x: size_mappings[x]['max_height'][0] if x in size_mappings else 0)

    # When the parent species is indicated instead of the child in 'traeart, 
    # use the mean of all trees in dataset with parent species as max_height
    species_no_height = gdf_wo_dup[gdf_wo_dup['max_height'] == 0]['traeart'].unique()
    for species_name in species_no_height:
        gdf_wo_dup.loc[(gdf_wo_dup['traeart'] == species_name) & (gdf_wo_dup['max_height'] == 0), 'max_height'] = gdf_wo_dup[(gdf_wo_dup['slaegt'] == species_name) & (gdf_wo_dup['max_height'] != 0)]['max_height'].mean()

    # Remove trees with no height information
    # i.e. species where only the 'slaegt' is nonempty, not 'traeart' for all trees
    gdf_wo_dup = gdf_wo_dup[gdf_wo_dup['max_height'] > 0]

    return gdf_wo_dup

def remove_tree(tree, borders, splits, min_dist):
    # Remove trees from one split that is within min_dist m of the borders of another split
    for split in splits:
        if split != tree['split']:
            dist = borders.loc[split]['geometry'].distance(tree['geometry'])
            if dist <= min_dist:
                return True
    return False

def split_trees(gdf_wo_dup):
    splits = ['train', 'val', 'test']
    neighs_split = [['Østerbro', 'Nørrebro', 'Amager Vest', 'Bispebjerg', 'Brønshøj-Husum', 'Vanløse'],
                    ['Amager Øst', 'Indre By'], 
                    ['Vesterbro-Kongens Enghave', 'Valby']]
    neigh_to_split = {neigh: split for neighs, split in zip(neighs_split, splits) for neigh in neighs}
    gdf_wo_dup['split'] = gdf_wo_dup['bydelsnavn'].map(neigh_to_split)

    # Load borders of neighborhoods
    borders = gpd.read_file('../data/raw/cph_borders.json')
    borders['split'] = borders['navn'].map(neigh_to_split)
    borders = borders.dissolve(by='split')
    borders.to_crs('epsg:25832', inplace=True)

    # Remove trees within min_dist of the border between two splits
    min_dist = 20
    gdf_wo_dup['remove'] = gdf_wo_dup.to_crs('epsg:25832').apply(lambda tree: remove_tree(tree, borders, splits, min_dist), axis=1)
    gdf_wo_dup.loc[gdf_wo_dup['remove'], ['split']] = None
    del gdf_wo_dup['remove']
    
    return gdf_wo_dup

if __name__ == "__main__":
    # Dict for renaming species to align with names on vdberk.com
    species_name_map = {"Acer campestre 'Queen Elisabeth'": "Acer campestre 'Queen Elizabeth'",
                        'Acer ginnala': 'Acer tataricum subsp. ginnala',
                        "Acer platanoides 'Farlakes Green'": "Acer platanoides 'Farlake's Green'",
                        "Acer saccharinum 'Pyramidialis'": "Acer saccharinum 'Pyramidale'",
                        "Aesculus hippoc. 'Baumannii'": "Aesculus hippocastanum 'Baumannii'",
                        "Amelanchier laevis 'Ballerina'": "Amelanchier 'Ballerina'",
                        'Cedrús atlántica': 'Cedrus libani subsp. atlantica',
                        'Crataegus lavallei': 'Crataegus lavalleei',
                        'Eleagnus angustifolia': 'Elaeagnus angustifolia',
                        "Fraxinus excelsior 'Pendula'": "Fraxinus excelsior 'Aurea Pendula'",
                        "Fraxinus pensylvanica 'Zundert'": "Fraxinus pennsylvanica 'Zundert'",
                        "Malus hybr. 'Rudolf'": "Malus 'Rudolph'",
                        "Malus hybr. 'Street Parade'": "Malus baccata 'Street Parade'",
                        "Malus x 'Elstar'": "Malus domestica 'Elstar'",
                        "Pinus sylvestris 'Typ Norwegen'": "Pinus sylvestris 'Norska'",
                        'Platanus hybr. acerifolia': 'Platanus hispanica',
                        "Populus simonii 'Fastgiata'": "Populus simonii 'Fastigiata'",
                        "Prunus ceracifera 'Nigra'": "Prunus cerasifera 'Nigra'",
                        "Prunus domestica 'Ouillins Reine Claude'": "Prunus domestica 'Reine Claude d'Oullins'",
                        "Prunus subhirtélla 'Accolade'": "Prunus 'Accolade'",
                        "Prunus subhirtella 'Hally Jolivette'": "Prunus 'Hally Jolivette'",
                        "Prunus x hilleri 'Spire'": "Prunus 'Spire'",
                        'Pyrus caucasica': 'Pyrus communis subsp. caucasica',
                        "Sorbus latifolia 'Atro'": "Sorbus latifolia 'Atrovirens'",
                        "Tilia cordata 'Green Spire'": "Tilia cordata 'Greenspire'",
                        'Tilia euchlora': 'Tilia europaea euchlora',
                        "Tilia tomentosa 'Brabrant'": "Tilia tomentosa 'Brabant'",
                        "Tilia x 'Zwarte Linde'": "Tilia europaea 'Zwarte Linde'",
                        "Sórbus commixta 'Dodong'": "Sorbus 'Dodong'",
                        'Styphnolobium japonica': 'Styphnolobium japonicum'}

    # Dict for adding manual height information
    # These are from various sources found by Googling the species name
    species_manual_height = {"Acer campestre": 17,
                             "Acer platanoides 'Allershausen'": 30,
                             "Acer platanoides 'Emerald Queen'": 25,
                             'Alnus spaethii': 18,
                             "Betula pendula 'Dalecarlica'": 20,
                             'Betula pendula fk penla': 30,
                             "Betula pubescens": 30,
                             "Crataegus crus-galli": 10,
                             'Crataegus intricata': 15,
                             "Crataegus laevigata": 10,
                             "Crataegus laevigata 'Paul's Scarlet'": 8,
                             "Crataegus lavallei": 12,
                             "Crataegus monogyna": 15,
                             "Crataegus monogyna 'PKP select'": 20,
                             'Crataegus prunifolia': 10,
                             "Fraxinus excelsior 'Robusta'": 25,
                             "Fraxinus pensylvanica 'Zundert'": 22,
                             "Juniperus communis 'Hibernica'": 6,
                             "Juniperus virginiana 'Sky Rocket'": 6,
                             "Laburnum watereri 'Vossii'": 10,
                             'Larix hybr. eurolepis': 30,
                             "Malus domestica 'Guldborg'": 10,
                             "Malus domestica 'Ildrød Pigeon'": 10,
                             "Malus domestica 'Rød Gråsten'": 6,
                             "Malus hybr. 'Akso'": 10,
                             "Malus hybr. 'Braendkjaer'": 15,
                             "Malus hybr. 'Butterball'": 10,
                             "Malus hybr. 'Dir. Moerland'": 10,
                             "Malus hybr. 'Red Sentinel'": 12,
                             'Malus sieboldii': 8,
                             "Malus toringo": 15,
                             'Pinus nigra': 15,
                             "Populus canadensis 'Bachalierii'": 40,
                             "Populus simonii 'Fastgiata'": 30,
                             'Prunus cerasifera': 9,
                             "Prunus cerasifera 'Nigra'": 12,
                             "Prunus avium 'Plena'": 20,
                             "Prunus domestica 'Incititia'": 5,
                             "Prunus padus 'Select'": 15,
                             "Prunus serrulata 'Shirofugen'": 6,
                             "Prunus subhirtellella 'Autumnalis": 12,
                             "Prunus umineko": 10,
                             'Pyrus calleryana': 18,
                             "Pyrus communis 'Beech Hill'": 17,
                             "Robinia pseudoacacia 'Bessoniana'": 25,
                             "Robinia pseudoacacia 'Monophylla'": 30,
                             "Robinia pseudoacacia 'Umbraculifera'": 25,
                             "Salix alba 'Liempde'": 35,
                             "Salix alba 'Saba'": 25,
                             "Salix alba 'Tristis'": 20,
                             "Salix caprea 'Mas'": 8,
                             "Salix matsudana 'Tortuosa'": 9,
                             'Sorbus hybrida': 15,
                             "Sorbus intermedia 'Annisse Kirke'": 12,
                             "Sorbus latifolia 'Atro'": 20,
                             'Sorbus mougeotii': 8,
                             "Tilia cordata 'Erecta'": 20,
                             "Tilia plathyphyllos 'Odin'": 20,
                             "Ulmus 'Rebona'": 25,
                             "Ulmus 'Regal'": 18}


    # Load and clean data
    gdf = load_data('../data/raw/tree_inventory.csv')
    gdf_wo_dup = remove_duplicates(gdf)
    gdf_wo_dup = remove_bushes(gdf_wo_dup)
    gdf_wo_dup_relevant = remove_irrelevant(gdf_wo_dup)

    # Get non-relevant trees for later use
    tree_ids_relevant = gdf_wo_dup_relevant['id'].unique()
    gdf_wo_dup_irrelevant = gdf_wo_dup[~gdf_wo_dup['id'].isin(tree_ids_relevant)]

    # Add general species info to trees
    gdf_wo_dup_relevant = get_species_info(gdf_wo_dup_relevant, species_name_map, species_manual_height)

    # Split trees
    gdf_wo_dup_relevant = split_trees(gdf_wo_dup_relevant)

    # Save cleaned data
    gdf_wo_dup_relevant.to_csv('../data/raw/tree_inventory_cleaned.csv', index=False)
    gdf_wo_dup_irrelevant.to_csv('../data/raw/tree_inventory_cleaned_irrelevant.csv', index=False)
