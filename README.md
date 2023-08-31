# Automatic Urban Street Tree Detection from Google Street Views using Faster R-CNN and Swin Transformer

<img width="738" alt="cover_photo" src="https://github.itu.dk/storage/user/1768/files/52023535-5850-46d1-be21-53b1fb233b5f">

This is the official repository for the paper *Automatic Urban Street Tree Detection from Google Street Views using Faster R-CNN and Swin Transformer*. The repository contains the following elements:

- **data:** The raw- and cleaned tree inventory, the Google Street Views and their metadata in panos.json, the annotated trees in street views in COCO format, the predicted annotations, the final predictions, and various mappings, e.g. from trees in the tree inventory to annotations and from predicted annotations to final predictions.
- **utils:** The file containing the main functions used in the project. 
- **dataset_creation:** Files for cleaning the tree inventory, extracting the street views, and annotating the street views.
- **exploratory_analysis:** Exploratory analysis of both the trees in the tree inventory and the annotated trees in images, and of the Faster R-CNN.
- **train:** The file used to train the Faster R-CNN with Swin Transformer and to fine-tune it to Pasadena Urban Trees.
- **models:** Trained models, both the original and the fine-tuned.
- **test:** Files for predicting bounding boxes in images and making final predictions on the Copenhagen and Pasadena datasets and for evaluating both.
