PlantVillage dataset is about 700 MB and can be downloaded from https://www.kaggle.com/datasets/emmarex/plantdisease

"dataset" directory should be like this format:

## Dataset

The `dataset/` directory contains the Plant Village dataset, which is organized into subdirectories for different plant categories and their respective conditions. Here's an example structure:

- **Pepper__bell___Bacterial_spot/**: This subdirectory contains images of pepper plants affected by bacterial spot disease. Sample images include:
  - `0022d6b7-d47c-4ee2-ae9a-392a53f48647___JR_B.Spot 8964.JPG`
  - `006adb74-934f-448f-a14f-62181742127b___JR_B.Spot 3395.JPG`
  - ...

- **Pepper__bell___healthy/**: This subdirectory contains images of healthy pepper plants. Sample images include:
  - `00100ffa-095e-4881-aebf-61fe5af7226e___JR_HL 7886.JPG`
  - `00208a93-7687-4e8c-b79e-3138687e0f38___JR_HL 7955.JPG`
  - ...

- ...

- **Tomato_healthy/**: This subdirectory contains images of healthy tomato plants. Sample images include:
  - `000146ff-92a4-4db6-90ad-8fce2ae4fddd___GH_HL Leaf 259.1.JPG`
  - `000bf685-b305-408b-91f4-37030f8e62db___GH_HL Leaf 308.1.JPG`
  - ...

Please note that the actual Plant Village dataset may have additional plant categories and condition subdirectories.

Make sure the images are named using unique identifiers followed by a descriptive label. This directory provides the necessary data for training, validation, or testing your models.



dataset/
    Pepper__bell___Bacterial_spot
        0022d6b7-d47c-4ee2-ae9a-392a53f48647___JR_B.Spot 8964.JPG
  	006adb74-934f-448f-a14f-62181742127b___JR_B.Spot 3395.JPG
	...
    
    Pepper__bell___healthy/
	00100ffa-095e-4881-aebf-61fe5af7226e___JR_HL 7886.JPG
	00208a93-7687-4e8c-b79e-3138687e0f38___JR_HL 7955.JPG
	


	
    ...
    Tomato_healthy/
	000146ff-92a4-4db6-90ad-8fce2ae4fddd___GH_HL Leaf 259.1.JPG
	000bf685-b305-408b-91f4-37030f8e62db___GH_HL Leaf 308.1.JPG
		
