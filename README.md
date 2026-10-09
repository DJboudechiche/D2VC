# D2VC
Deep distributed video coding

Download the dataset (Vimeo90K) : http://data.csail.mit.edu/tofu/dataset/vimeo_triplet.zip

coded the first and the third frame of Vimeo90K dataset by intracoder (MLIC++ or intra H.266/VVC)

Use the first and the third frames to create (offline) the side information frame using the interpolation model (CDFI model)

Train the model D2VC using the second frame and the side information frame using mainColor.py

Put into output/weight/... the weight of D2VC model and test the code using test.py   

The pretrained models will be made publicly available on Github upon publication of the paper to ensure reproducibility and facilitate further research. 


