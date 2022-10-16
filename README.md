# AI-assisted tool for evaluating aesthetic and composition quality of blender 3D scenes
Personal project for the AI & Society minor by Vladimir Vladinov

## Generating training data
With the add-on you can create your own random labelled data to train the algorithm on. This can be done with the "Generate" sub-menu
![image](https://user-images.githubusercontent.com/22458048/196052012-24b89356-766d-433f-9862-cda2ee828d49.png)

| Icon  | Name | Description |
| ------------- | ------------- | ------------- |
| ![image](https://user-images.githubusercontent.com/22458048/196052060-ff9d67b6-50dc-4b15-b165-a4ad5a79e296.png)  | Label | Content of the scene that's about to be generated |
| ![image](https://user-images.githubusercontent.com/22458048/196052118-393e0a5a-1128-48fb-9334-0e4fc1a9a358.png) | Objects to be generated | The amount of objects that will be present in the scene
| ![image](https://user-images.githubusercontent.com/22458048/196052160-4eb9e4b0-68f3-4c18-acc2-fd6265ae13be.png) | Ground generation | Toggle the generation of a ground plane |
| ![image](https://user-images.githubusercontent.com/22458048/196052243-ce911060-2aad-481b-aec7-b9dc4f52aa7d.png) | Generate scene | Generate the scene with the given parameters |
| ![image](https://user-images.githubusercontent.com/22458048/196052276-9fab8a9f-6891-4a61-bb82-568ac0afd41e.png) | Ground bias | Amount of objects to be placed on the ground, if the button is toggled the value is randomized for every scene |
| ![image](https://user-images.githubusercontent.com/22458048/196052397-ba14b3e3-9bda-468f-8748-891bca433168.png) | Rating | Rate the generated scene on a 1-5 scale |
| ![image](https://user-images.githubusercontent.com/22458048/196052426-d3446dda-5e13-4ef8-ae13-1a6013a184e9.png) | Render | Render the scene and save it with an appropriate name, which includes the rating, amount of objets and whether a ground plane is present |
| ![image](https://user-images.githubusercontent.com/22458048/196052479-f2e48e60-a4ef-431c-b231-d9b8571181fd.png) | Ouput Path | Folder for saving the generated frames |

Examples of images generated:

![4_05_G](https://user-images.githubusercontent.com/22458048/196052587-ac042da8-37e8-44d3-ac59-b8033882f48c.png)
![2_05_N](https://user-images.githubusercontent.com/22458048/196052591-94b14fb3-cd83-464a-a905-3ff77a1aae2f.png)
![3_10_G](https://user-images.githubusercontent.com/22458048/196052595-6321a5e5-1eb8-44b0-b162-f04dff1e41ac.png)
![5_10_N](https://user-images.githubusercontent.com/22458048/196052618-8f95e778-298c-4bda-be4c-11d90d1fdd1c.png)
