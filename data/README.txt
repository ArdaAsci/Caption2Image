data/encoded_test_captions.pt --------------> Test captions encoded by CLIP model as torch.Tensor
data/encoded_test_X.pt ---------------------> Test images encoded by CLIP model as torch.Tensor
data/encoded_train_captions.pt -------------> Train captions encoded by CLIP model as torch.Tensor
data/healty_test_urls.npy ------------------> Indicates which test urls are healthy as numpy.ndarray
data/healty_train_urls.npy -----------------> Indicates which train urls are healthy as numpy.ndarray
data/test_image_features.pt ----------------> Test image features encoded by CLIP model as torch.Tensor
data/test_X.npy ----------------------------> Final generated test features (cleared of missing urls) as numpy.ndarray
data/test_Y.npy ----------------------------> Final generated test image indices (cleared of missing urls) as numpy.ndarray
data/tokenized_test_captions.pt ------------> Test captions tokenized by CLIP model as torch.Tensor (not required if encoded_test_captions.pt exists)
data/tokenized_train_captions.pt -----------> Train captions tokenized by CLIP model as torch.Tensor (not required if encoded_train_captions.pt exists)
data/train_image_features.pt ---------------> Train image features encoded by CLIP model as torch.Tensor
data/train_X.npy ---------------------------> Final generated train features (cleared of missing urls) as numpy.ndarray
data/train_Y.npy ---------------------------> Final generated train image indices (cleared of missing urls) as numpy.ndarray
data/validation_X.npy ----------------------> Final generated validation features (cleared of missing urls) as numpy.ndarray
data/validation_Y.npy ----------------------> Final generated validation image indices (cleared of missing urls) as numpy.ndarray
