# Pedestrian classification

## Authors
 - [Erik Matovič](https://github.com/Matovic)
 - Jakub Horvat 

## Solution
We use the PIE class from [pie_data Python file](./src/pie_data.py) to make our train, validation and test dataset. Python file pie_data is available with XML annotations to extract labels.

### 1. Exploratory Data Analysis & Data Transformations
Frame from the [PIE dataset videos](https://data.nvision2.eecs.yorku.ca/PIE_dataset/):
 <p align="center">
	<img src="./outputs/data.png">
</p>

To extract and save only annotated frames(roughly 1TB) from video sets, we use the PIE object method:

```python3
def extract_img(pie: PIE) -> None:
    """
    Extract and save only annotated frames.
    :param: pie: PIE object
    """
    pie.extract_and_save_images(extract_frame_type='annotated')
```

We make three instances of our defined class to extract only annotated pedestrians. Training, validation, and test sets contain 139 404 frames, 43 885 frames and 109 641 frames, respectively.

```python3
class PIE_peds():
    def __init__(self, setname: str, pie: PIE) -> None:
        #super().__init__()
        self.setname = setname
        self.set_dir = dict()
        self.all_path = list()
        self.all_filenames = list()
        self.path = pie._images_path

        assert setname in ['train', 'val', 'test']
        
        self.img_init(pie)
        
    
    def __len__(self):
        return len(self.all_filenames)
    
    def img_init(self, pie) -> None:
        set_list = pie._get_image_set_ids(self.setname)

        for set_ in set_list:
            set_path = join(pie._images_path, set_)
            self.set_dir[set_] = dict()
            
            set_content = listdir(set_path)
            for video in set_content:
                video_path = join(set_path, video)
                imgs = listdir(video_path)
                self.all_path += ([video_path] * len(imgs))
                self.all_filenames += imgs
                self.set_dir[set_][video] = imgs
        print(self.set_dir)    
```

Next, we make our image dataset with CSV annotations. It is also necessary to encode gender and crossing values for our architecture. For females, it is 0, and for irrelevant crossing, it is 2. 

```python3
def save_img_annotations(dataset: PIE_peds, folder: str) -> None:
    """
    Save images into given folder and also CSV annotations. 

    :param dataset: PIE_peds class, where all paths are set. 
    :param folder: folder train or val or test to save pedestrians.
    """
    assert folder in ['train', 'val', 'test']

    target = dict()
    target['set'] = []
    target['video'] = []
    target['frame'] = []
    target['ped_idx'] = []
    target['ped'] = []
    target['BBox'] = []
    target['action'] = []
    target['age'] = []
    target['gender'] = []
    target['look'] = []
    target['cross'] = []

    for set_name in dataset.set_dir:
        for video in dataset.set_dir[set_name]:
            annotations = pie._get_annotations(set_name, video)
            annotations_attributes = pie._get_ped_attributes(set_name, video)
            for frame in dataset.set_dir[set_name][video]: 
                img_path = '../images/' + set_name + '/' + video + '/' + frame
                video + '/' + frame

                img = cv2.imread(img_path)
                for idx in annotations['ped_annotations']:
                    frame_idx = int(frame[:-4])
                    # get only annotated frames with pedestrians
                    if frame_idx in annotations['ped_annotations'][idx]['frames']:
                        # skip existing pedestrian
                        if idx in target['ped']:
                            break

                        frame_key = annotations['ped_annotations'][idx]['frames'].index(frame_idx)
                        
                        # set annotations in dictionary
                        target['set'].append(set_name)
                        target['video'].append(video)
                        target['frame'].append(frame)
                        target['ped_idx'].append(frame_key)
                        target['ped'].append(idx)
                        target['BBox'].append(annotations['ped_annotations'][idx]['bbox'][frame_key])
                        target['action'].append(annotations['ped_annotations'][idx]['behavior']['action'][frame_key])
                        target['age'].append(annotations_attributes[idx]['age'])
                        target['gender'].append(annotations_attributes[idx]['gender'])
                        target['look'].append(annotations['ped_annotations'][idx]['behavior']['look'][frame_key])
                        target['cross'].append(annotations['ped_annotations'][idx]['behavior']['cross'][frame_key])
                        
                        # BBox for pedestrian
                        x1 = floor(target['BBox'][-1][0])
                        y1 = floor(target['BBox'][-1][1])
                        x2 = ceil(target['BBox'][-1][2])
                        y2 = ceil(target['BBox'][-1][3])

                        # crop pedestrian and make grayscale
                        crop_img = img[y1:y2, x1:x2]
                        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                        crop_img = cv2.resize(crop_img, (64, 64))
                        
                        # save image
                        f_name = '../images/' + folder + '/' + set_name + '_' + video + '_' + target['ped'][-1] + '_' + frame
                        cv2.imwrite(f_name, crop_img)

    # save annotations as CSV file                    
    df = pd.DataFrame(data=target)
    annotations_path = '../' + folder + '_annotations.csv'
    df.to_csv(annotations_path)
```

Exploratory data analysis shows the imbalanced dataset. As a result, we cloned training and validation images with low crossing and looking values and augmented cloned frames by rotating them. We also append our annotations with augmented clones. Furthermore, we build our architecture without the age class because age values are the most imbalanced.
<p align="center">
	<img src="./outputs/ped_action.png">
	<img src="./outputs/ped_age.png">
	<img src="./outputs/ped_cross.png">
	<img src="./outputs/ped_gender.png">
	<img src="./outputs/ped_look.png">
</p>

```python3
df = pd.DataFrame() # we append our annotations with augmented clones

root_path = '../images/train/'  # <root path for train or val>
for file in imgs_names:
    img_path = root_path + file
    img = Image.open(Path(img_path))

    # for train, for val it is 14 and 36
    set_idx_start = 16
    ped_index_start = 38

    # pedestrian index is varying
    ped_index = img_path.index('_', ped_index_start)

    set_name = img_path[set_idx_start:set_idx_start+5]
    video_name = img_path[set_idx_start+6:set_idx_start+16]
    ped_name = img_path[set_idx_start+17:ped_index]
    frame_name = img_path[ped_index + 1:ped_index + 10]

    for i in range(10):
        degree = int((torch.rand(1) * 360) % 360)
        rotated_img = transforms.RandomRotation(degrees=degree)(img)
        # save a image using extension
        fname = img_path + f'_rotated_{i}.png'

        # df_clones is a pandas dataframe containg cross values to be cloned
        # later it contains look values to be cloned 
        label = df_clones.loc[
            (df_clones.set == set_name) &
            (df_clones.video == video_name) &
            (df_clones.frame == frame_name) &
            (df_clones.ped == ped_name), :
        ]
        
        df2 = label.loc[label.frame == frame_name, :]
        df2['frame'] = frame_name + f'_rotated_{i}.png'
        df = pd.concat([df, df2])
        rotated_img = rotated_img.save(fname)
```
Initially, the training dataset contains 703 different pedestrians. We then have 1993 training images by cloning and rotating clones.
 <p align="center">
	<img src="./outputs/ped_action_update.png">
	<img src="./outputs/ped_cross_update.png">
	<img src="./outputs/ped_gender_update.png">
	<img src="./outputs/ped_look_update.png">
</p>

We made our pedestrian dataset with annotations from the PIE dataset with warped grayscale images with dimensions 64x64. Final transformed data can be downloaded from [here](https://stubask-my.sharepoint.com/:u:/g/personal/xmatovice_stuba_sk/EbJaQifX48pEg9o0ZDGJ-ewB79fffZ8ATrQ1ylEZh3EbsQ?e=HlRs85):
 <p align="center">
	<img src="./outputs/input.png">
</p>

### 2. Data Preprocessing
The PIE dataset is split into six sets, where training data comprise set01, set02 and set04. Set05 and set06 made up validation data, and test data contain set03. We created our dataset class which gets items by their index and returns input data X, four target variables y_action, y_gender, y_look, y_cross and image index:

```python3
class PIE_dataset(Dataset):
    """
    Dataset class for dataloader.
    """
    def __init__(self, setname: str, pie: PIE) -> None:
        """
        Dataset init.
        :param setname: specifying trainig, validation or test set
        """
        assert setname in ['train', 'val', 'test'], 'wrong setname, accepting only \'train\', \'val\', \'test\''
        
        super().__init__()
        self.setname = setname
        self.img_path = pie._images_path + '/' + self.setname + '/'
        self.annotations_path = '../' + self.setname + '_annotations.csv'
        self.all_filenames = listdir(self.img_path)
        self.all_annotations = pd.read_csv(self.annotations_path)     
        self.all_annotations_meaning = self.all_annotations.columns.values.tolist()

        # Setting labels
        self.label_action = self.all_annotations['action']
        self.label_gender = self.all_annotations['gender']
        self.label_look = self.all_annotations['look']
        self.label_cross = self.all_annotations['cross']   
        
    
    def __len__(self) -> int:
        """
        Return the total number of images.
        returns: The total number of images.
        """
        return len(self.all_filenames)
    

    def __getitem__(self, index) -> dict:
        """
        Get item with annotations.
        :param index: the number of image
        returns: Dictionary.
        """
        file_name = self.all_filenames[index]
        img_path = self.img_path + file_name
        
        # Read the input image
        img = Image.open(img_path)#.convert('RGB')
        # convert image to torch tensor
        img_tensor = transforms.ToTensor()(img)
        # transform to normalize the image with mean and std
        transform = transforms.Normalize(mean=(0.2,), std=(0.2,))
        normalized_img_tensor = transform(img_tensor)

        # train
        set_idx_start = 16
        ped_index_start = 38
        
        # val
        if self.setname == 'val':
            set_idx_start = 14
            ped_index_start = 36

        # test
        if self.setname == 'test':
            set_idx_start = 15
            ped_index_start = 37

        # pedestrian index is varying from index 38 to 41
        ped_index = img_path.index('_', ped_index_start)

        set_name = img_path[set_idx_start:set_idx_start+5]
        video_name = img_path[set_idx_start+6:set_idx_start+16]
        ped_name = img_path[set_idx_start+17:ped_index]
        frame_name = img_path[ped_index + 1:]
        
        label = self.all_annotations.loc[
            (self.all_annotations.set == set_name) &
            (self.all_annotations.video == video_name) &
            (self.all_annotations.frame == frame_name) &
            (self.all_annotations.ped == ped_name)
        ]

        label_action = torch.tensor(self.label_action[label.index[0]], dtype=torch.float)
        label_gender = torch.tensor(self.label_gender[label.index[0]], dtype=torch.float)
        label_look = torch.tensor(self.label_look[label.index[0]], dtype=torch.float)
        label_cross = torch.tensor(self.label_cross[label.index[0]], dtype=torch.long)
        
        return {'data': normalized_img_tensor,
                'label_action': label_action,
                'label_gender': label_gender,
                'label_look': label_look,
                'label_cross': label_cross,
                'img_idx': index}
```

### 3. Model
Best parameters from WandB:
 - batch size: 32,
 - hidden size: 256,
 - epochs: 50,
 - learning rate: 0.05657808132757078
 - momentum: 0.7500727372948774

Our convolutional neural network is defined with two convolutional layers, each followed by Leaky ReLU activations. After CNN layers, we have a pooling layer with kernel size 2x2 to get only max values. Six fully connected dense layers are followed after the polling layer. Each dense layer also has the same activation function as CNN layers, but we added dropout with a 20% likelihood to zeroed an element. In the end, the feed-forward pass returns four classes - standing or walking pedestrian, pedestrian's gender, pedestrian looking at the camera or not, and pedestrian crossing the road. 

We use the Binary Cross Entropy Loss function with a Sigmoid layer for the first three binary classes. The last class is a multiclass prediction; therefore, we use the Cross-Entropy Loss function with logSoftMax. Stochastic Gradient Descent with Momentum is used as an optimization algorithm.

```python3
class CNN(nn.Module):
    """
    Model class
    """
    def __init__(self, n_channels, n_features) -> None:
        """
        init
        :param n_channels: number of input challens
        """
        super(CNN, self).__init__()
        self.n_channels = n_channels
        self.n_features = n_features

        self.conv11 = nn.Conv2d(in_channels=n_channels, out_channels=64, 
                             kernel_size=(3, 3), padding=0, dilation=2)
        self.conv12 = nn.Conv2d(in_channels=64, out_channels=64,
			    kernel_size=(3, 3), padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # MLP
        self.fc1 = nn.Linear(in_features=53824, out_features=n_features)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.2)

        self.fc2 = nn.Linear(in_features=n_features, out_features=n_features)
        self.fc3 = nn.Linear(in_features=n_features, out_features=n_features)
        self.fc4 = nn.Linear(in_features=n_features, out_features=2*128)
        self.fc5 = nn.Linear(in_features=2*128, out_features=2*128)

        self.fc_action = nn.Linear(2*128, 1)      # output action class
        self.fc_gender = nn.Linear(2*128, 1)      # output gesture class
        self.fc_look = nn.Linear(2*128, 1)        # output look class
        self.fc_cross = nn.Linear(2*128, 3)       # output cross class

        self.logSoftmax = nn.LogSoftmax(dim=1)
        self.BCELoss = nn.BCEWithLogitsLoss() 


    def forward(self, x) -> dict:        
        """
        forward pass
        :param x: data x
        """
        output = self.conv11(x)
        output = self.relu(output)
        output = self.conv12(output)
        output = self.relu(output)
        output = self.maxpool(output)

		# flatten the output from the previous layer 
        output = flatten(output, 1)
        output = self.fc1(output)
        output = self.relu(output)
        output = self.dropout(output)
		
        output = self.fc2(output)
        output = self.relu(output)
        output = self.dropout(output)

        output = self.fc3(output)
        output = self.relu(output)
        output = self.dropout(output)

        output = self.fc4(output)
        output = self.relu(output)
        output = self.dropout(output)

        output = self.fc5(output)
        output = self.relu(output)
        output = self.dropout(output)

        # pass output to 4 different layers to get 4 classes
        label_action = self.fc_action(output)
        label_gender = self.fc_gender(output) # torch.sigmoid(self.fc2(X))  
        label_look = self.fc_look(output)
        label_cross = self.fc_cross(output)
        
        # return 4 classes
        return {'label_action': label_action,
                'label_gender': label_gender,
                'label_look': label_look,
                'label_cross': label_cross}

model = CNN(
    n_channels=n_channels, 
    n_features=n_features
) 
model.to(device)

opt = optim.SGD(model.parameters(), lr=INIT_LR, momentum=MOMENTUM)
lossFn = nn.CrossEntropyLoss()
```

### 4. Training & validation
Training and validation were done with early stopping monitoring validation loss.

```python3
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() 
    return correct


def val_cnn(val_dl, loss_fn, model, device):
    """
    Validation
    """
    # init epoch validation counters
    epoch_train_accuracy_action, epoch_train_total_action, \
        epoch_train_true_action, epoch_train_loss = 0, 0, 0, 0
    
    epoch_train_accuracy_gender, epoch_train_total_gender, \
        epoch_train_true_gender = 0, 0, 0
    
    epoch_train_accuracy_look, epoch_train_total_look, \
        epoch_train_true_look = 0, 0, 0
    
    epoch_train_accuracy_cross, epoch_train_total_cross, \
        epoch_train_true_cross = 0, 0, 0
    
    # disable gradient calculation
    with torch.no_grad():
        # enumerate mini batches
        for _, sample in enumerate(val_dl):
            # get X and y with index from sample
            X_batch, y_batch_action, y_batch_gender, \
                y_batch_look, y_batch_cross, idx = sample['data'], sample['label_action'], \
                    sample['label_gender'], sample['label_look'], sample['label_cross'], \
                        sample['img_idx']
            
            X_batch, y_batch_action, y_batch_gender, y_batch_look, y_batch_cross = \
                X_batch.to(device), y_batch_action.to(device), y_batch_gender.to(device), \
                    y_batch_look.to(device), y_batch_cross.to(device)
                     
            # compute the model output
            y_hat = model(X_batch)
            y_action = y_hat['label_action']
            y_gender = y_hat['label_gender']
            y_look = y_hat['label_look']
            y_cross = y_hat['label_cross']

            y_cross_ = model.logSoftmax(y_cross) 
            y_cross_1 = torch.argmax(y_cross_, dim=1).type(torch.LongTensor).to(device)       
            _, y_action_ = torch.max(y_action, dim=1) 
            _, y_gender_ = torch.max(y_gender, dim=1)
            _, y_look_ = torch.max(y_look, dim=1)

            #y_cross_1 = torch.argmax(y_cross)
            loss_action = model.BCELoss(y_action, y_batch_action.unsqueeze(1))
            loss_gender = model.BCELoss(y_gender, y_batch_gender.unsqueeze(1))
            loss_look = model.BCELoss(y_look, y_batch_look.unsqueeze(1))
            loss_cross = loss_fn(y_cross_, y_batch_cross)
            
            loss = loss_action + loss_gender + loss_look + loss_cross
            
            # update train counters
            epoch_train_loss += loss.item()

            epoch_train_true_action += accuracy_fn(y_action_, y_batch_action)
            epoch_train_total_action += len(y_batch_action)

            epoch_train_true_gender += accuracy_fn(y_gender_, y_batch_gender)
            epoch_train_total_gender += len(y_batch_gender)

            epoch_train_true_look += accuracy_fn(y_look_, y_batch_look)
            epoch_train_total_look += len(y_batch_look)

            epoch_train_true_cross += accuracy_fn(y_cross_1, y_batch_cross)
            epoch_train_total_cross += len(y_batch_cross)

        # update train accuracy & loss statistics
        epoch_train_loss /= (len(val_dl.dataset)/val_dl.batch_size)
        
        epoch_train_accuracy_action = (epoch_train_true_action/epoch_train_total_action) * 100
        epoch_train_accuracy_gesture = (epoch_train_true_gender/epoch_train_total_gender) * 100
        epoch_train_accuracy_look = (epoch_train_true_look/epoch_train_total_look) * 100
        epoch_train_accuracy_cross = (epoch_train_true_cross/epoch_train_total_cross) * 100

    return epoch_train_loss, epoch_train_accuracy_action, epoch_train_accuracy_gesture, \
        epoch_train_accuracy_look, epoch_train_accuracy_cross

def train_cnn(train_dl:DataLoader, val_dl:DataLoader, n_epochs:int, optimizer: optim, model: nn.Module, loss_fn: nn.NLLLoss, device):
    """
    Training
    """
    # init train lists for statistics
    loss_train, acc_action_train, acc_gender_train, \
        acc_look_train, acc_cross_train = list(), list(), list(), list(), list()

    # init validation lists for statistics
    loss_val, acc_action_val, acc_gender_val, \
        acc_look_val, acc_cross_val = list(), list(), list(), list(), list()
    
    early_stopper = EarlyStopper(patience=3, min_delta=1)    
    # enumerate epochs
    for epoch in range(n_epochs):
        # init epoch train counters
        epoch_train_accuracy_action, epoch_train_total_action, \
            epoch_train_true_action, epoch_train_loss = 0, 0, 0, 0
        
        epoch_train_accuracy_gender, epoch_train_total_gender, \
            epoch_train_true_gender = 0, 0, 0
        
        epoch_train_accuracy_look, epoch_train_total_look, \
            epoch_train_true_look = 0, 0, 0
        
        epoch_train_accuracy_cross, epoch_train_total_cross, \
            epoch_train_true_cross = 0, 0, 0

        # enumerate mini batches
        for _, sample in enumerate(train_dl):
            # get X and y with index from sample
            X_batch, y_batch_action, y_batch_gender, \
                y_batch_look, y_batch_cross, idx = sample['data'], sample['label_action'], \
                    sample['label_gender'], sample['label_look'], sample['label_cross'], \
                        sample['img_idx']
            
            X_batch, y_batch_action, y_batch_gender, y_batch_look, y_batch_cross = \
                X_batch.to(device), y_batch_action.to(device), y_batch_gender.to(device), \
                    y_batch_look.to(device), y_batch_cross.to(device)
            
            # clear the gradients
            optimizer.zero_grad()
            
            # compute the model output
            y_hat = model(X_batch)
            y_action = y_hat['label_action']
            y_gender = y_hat['label_gender']
            y_look = y_hat['label_look']
            y_cross = y_hat['label_cross']

            y_cross_ = model.logSoftmax(y_cross) 
            y_cross_1 = torch.argmax(y_cross_, dim=1).type(torch.LongTensor).to(device)
            
            _, y_action_ = torch.max(y_action, dim=1) #.round().int()
            _, y_gender_ = torch.max(y_gender, dim=1) #.round().int()
            _, y_look_ = torch.max(y_look, dim=1) #.round().int()
            
            loss_action = model.BCELoss(y_action, y_batch_action.unsqueeze(1))
            loss_gender = model.BCELoss(y_gender, y_batch_gender.unsqueeze(1))
            loss_look = model.BCELoss(y_look, y_batch_look.unsqueeze(1))
            loss_cross = loss_fn(y_cross_, y_batch_cross)

            loss = loss_action + loss_gender + loss_look + loss_cross
            
            loss.backward()
            
            optimizer.step()

            # update train counters
            epoch_train_loss += loss.item()

            epoch_train_true_action += accuracy_fn(y_action_, y_batch_action)
            epoch_train_total_action += len(y_batch_action)

            epoch_train_true_gender += accuracy_fn(y_gender_, y_batch_gender)
            epoch_train_total_gender += len(y_batch_gender)

            epoch_train_true_look += accuracy_fn(y_look_, y_batch_look)
            epoch_train_total_look += len(y_batch_look)

            epoch_train_true_cross += accuracy_fn(y_cross_1, y_batch_cross)
            epoch_train_total_cross += len(y_batch_cross)

        # update train accuracy & loss statistics
        epoch_train_loss /= (len(train_dl.dataset)/train_dl.batch_size)
        
        epoch_train_accuracy_action = (epoch_train_true_action/epoch_train_total_action) * 100
        epoch_train_accuracy_gender = (epoch_train_true_gender/epoch_train_total_gender) * 100
        epoch_train_accuracy_look = (epoch_train_true_look/epoch_train_total_look) * 100
        epoch_train_accuracy_cross = (epoch_train_true_cross/epoch_train_total_cross) * 100
        
        # validation
        epoch_val_loss, epoch_val_acc_action, epoch_val_acc_gender, \
            epoch_val_acc_look, epoch_val_acc_cross = val_cnn(val_dl=val_dl, 
                                                              model=model, 
                                                              loss_fn=loss_fn, 
                                                              device=device)
        
        # update global train stats
        loss_train.append(epoch_train_loss)
        acc_action_train.append(epoch_train_accuracy_action)
        acc_gender_train.append(epoch_train_accuracy_gender)
        acc_look_train.append(epoch_train_accuracy_look)
        acc_cross_train.append(epoch_train_accuracy_cross)

        # update global validation stats
        loss_val.append(epoch_val_loss)
        acc_action_val.append(epoch_val_acc_action) 
        acc_gender_val.append(epoch_val_acc_gender)
        acc_look_val.append(epoch_val_acc_look)
        acc_cross_val.append(epoch_val_acc_cross)

        # print
        if epoch % 1 == 0: #== (n_epochs - 1):
            print(
                f'Epoch {epoch}/{n_epochs}: \
                train loss {loss_train[-1]}, \
                val loss {loss_val[-1]}, \
                action train acc {acc_action_train[-1]}, \
                gender train acc {acc_gender_train[-1]}, \
                look train acc {acc_look_train[-1]}, \
                cross train acc {acc_cross_train[-1]}, \
                action val acc {acc_action_val[-1]}, \
                gender val acc {acc_gender_val[-1]}, \
                look val acc {acc_look_val[-1]}, \
                cross val acc {acc_cross_val[-1]}'
            )

        # early stopping
        if early_stopper.early_stop(epoch_val_loss):    
            print(f'Early stopped at {epoch}')         
            break
        
    return loss_train, acc_action_train, acc_gender_train, acc_look_train, \
        acc_cross_train, loss_val, acc_action_val, acc_gender_val, \
            acc_look_val, acc_cross_val
```

Tracking training loss is done with Weights and Biases. Training loss is decreasing, signalling the model is learning on the training data.
<p align="center">
	<img src="./outputs/train loss.png">
</p>

Validation loss after a few epochs starts increasing until the early stopping. Meaning the model is overfitting on the training data.
<p align="center">
	<img src="./outputs/val loss.png">
</p>

Training and validation accuracy of pedestrian crossing classification:
<p align="center">
	<img src="./outputs/cross train acc.png">
	<img src="./outputs/cross val acc.png">
</p>

Validation accuracy of pedestrian action classification:
<p align="center">
	<img src="./outputs/action val acc.png">
</p>

Training and validation loss run with the best parameters.
<p align="center">
	<img src="./outputs/train_val_loss.png">
</p>

Training and validation accuracy of pedestrian crossing classification. Run with the best parameters:
<p align="center">
	<img src="./outputs/train_val_acc_cross.png">
</p>

### 5. Testing

```python3
def evaluation(model, test_dl):
    """
    evaluation
    """    
    y_test_all_action, y_test_all_gender, y_test_all_cross, y_test_all_look = list(), list(), list(), list()
    y_all_action, y_all_gender, y_all_cross, y_all_look = list(), list(), list(), list()
    # total_action, total_gesture, total_cross, total_look = 0, 0, 0, 0
    correct_action, correct_gender, correct_cross, correct_look = 0, 0, 0, 0
    # disable gradient calculation
    with torch.no_grad():
        # enumerate mini batches
        for _, sample in enumerate(test_dl):
            # get X and y with index from sample
            X_batch, y_batch_action, y_batch_gender, \
                y_batch_look, y_batch_cross, idx = sample['data'], sample['label_action'], \
                    sample['label_gender'], sample['label_look'], sample['label_cross'], \
                        sample['img_idx']
            
            X_batch, y_batch_action, y_batch_gender, y_batch_look, y_batch_cross = \
                X_batch.to(device), y_batch_action.to(device), y_batch_gender.to(device), \
                    y_batch_look.to(device), y_batch_cross.to(device)
                     
            # compute the model output
            # Make prediction logits with model
            y_hat = model(X_batch)
            y_action = y_hat['label_action']
            y_gender = y_hat['label_gender']
            y_look = y_hat['label_look']
            y_cross = y_hat['label_cross']
            
            y_cross_ = model.logSoftmax(y_cross) 
            true_cross_max = torch.argmax(y_cross_, dim=1)

            _, y_action_ = torch.max(y_action, dim=1) #.round().int()
            _, y_gender_ = torch.max(y_gender, dim=1) #.round().int()
            _, y_look_ = torch.max(y_look, dim=1) #.round().int()
            
            # accuracies
            true_action = accuracy_fn(y_action_, y_batch_action)
            true_gender = accuracy_fn(y_gender_, y_batch_gender)
            true_look = accuracy_fn(y_look_, y_batch_look)
            true_cross = accuracy_fn(true_cross_max, y_batch_cross)
        
            # update predictions stats
            y_all_action.extend(y_action_.cpu().numpy())
            y_all_gender.extend(y_gender_.cpu().numpy())
            y_all_look.extend(y_look_.cpu().numpy())
            y_all_cross.extend(true_cross_max.cpu().numpy())

            # update batch y stats
            y_test_all_action.extend(y_batch_action.cpu().numpy())
            y_test_all_gender.extend(y_batch_gender.cpu().numpy())
            y_test_all_look.extend(y_batch_look.cpu().numpy())
            y_test_all_cross.extend(y_batch_cross.cpu().numpy())

            correct_action += true_action
            correct_gender += true_gender
            correct_look += true_look
            correct_cross += true_cross
            
    report_action = classification_report(y_test_all_action, y_all_action, target_names=['0', '1'])
    report_gender = classification_report(y_test_all_gender, y_all_gender, target_names=['0', '1'])
    report_look = classification_report(y_test_all_look, y_all_look, target_names=['0', '1'])
    report_cross = classification_report(y_test_all_cross, y_all_cross, target_names=['0', '1', '2'])

    matrix_action = confusion_matrix(y_test_all_action, y_all_action)
    matrix_gender = confusion_matrix(y_test_all_gender, y_all_gender)
    matrix_look = confusion_matrix(y_test_all_look, y_all_look)
    matrix_cross = confusion_matrix(y_test_all_cross, y_all_cross)
    
    matrix_action_display = ConfusionMatrixDisplay(matrix_action, display_labels=['0', '1'])
    matrix_gender_display = ConfusionMatrixDisplay(matrix_gender, display_labels=['0', '1'])
    matrix_look_display = ConfusionMatrixDisplay(matrix_look, display_labels=['0', '1'])
    matrix_cross_display = ConfusionMatrixDisplay(matrix_cross, display_labels=['0', '1', '2'])

    return report_action, report_gender, report_look, report_cross, \
        matrix_action_display, matrix_gender_display, \
            matrix_look_display, matrix_cross_display
```

<p align="center">
	<img src="./outputs/classification_action.png">
	<img src="./outputs/classification_gender.png">
	<img src="./outputs/classification_look.png">
	<img src="./outputs/classification_cross.png">
</p>


## Conclusion
We have utilized our multi-output classification on the GPU using CUDA. However, the results are not satisfying. Our made-up dataset from the PIE dataset has various problems—imbalanced classes, possibly small images or unrecognizable pedestrians. We have tried to address the class imbalance. Nonetheless, we could not bring our dataset to equal balance because by balancing one label from one class, other labels from three classes emerged.   

Validation loss after a few epochs always started increasing, and later it was stopped by an early stopping. Meaning our classifier began to be overfitted on the training data.

## Changelog
05.04.2023 - spravené 
 - Exploratory Data Analysis 
 - Rozframeovanie videí na snímky 
    - Iba anotované snímky nakoľko celý dataset by mal cca 3TB 
    - Vytiahnuté niekoľkých anotovaných snímok(zhruba 1000) nakoľko všetky frame-y majú cca 1TB 
    - Vytiahnutie behaviorálnych anotácií chodcov z XML a spravenie CSV súboru z toho 
        - Action - stojí alebo chodí, triedy 0-1 
        - Gesture – pohyby rúk, hlavy alebo iné, triedy: 0-5 
        - Look - pozerá sa do kamery alebo nie, triedy: 0,1 
        - Cross – ide cez prechod alebo nie alebo je to irelvantné, triedy: 0,1,2 
 - Načítanie datasetu a anotácií 
 - Rozdelenie datasetu na trénovací, validačný a testovací 
 - Návrh architektúry CNN 
    - Inšpirácia VGG 
 - PyTorch 

05.04.2023 - TO DO 
 - Moznosti riesenia:  
    - pomocou regresiu  
    - rozvetvenu siet a potom z nich dalsie mlp 
    - mozno aj rezidualne bloky do siete 
 - je potrebne predspracovanie obrazkov - padding, warp, upscaling 
 - Mozno aj do grayscale, mozno aj augmentovat obrazky - pre zvacsenie datasetu 
 - Vytiahnuť chodcov pomocou Boundig Boxov a tá časť obrázkov pôjde na vstup CNN a toho chodca budeme klasifikovať do tried action, gesture, look a cross 
 - Tréning 
 - Validácia 
 - Testovanie 
 - Early stopping 
 - WandB 

12.04.2023 - spravené 
 - Rozbehanie WandB a vygenerovanie pdf report 
 - EDA - dokončenie 
    - Výber 4 finálnych tried kvôli nevyváženosti dátam 
    - Pridanie grafov do dokumentácie 
 - Data Transformation 
    - Videa na frames 
    - Rozdelenie frames na train-dev-test 
    - Extrahovanie unikátnych chodcov zo snímok pomocou BBoxov, unikátnych preto, aby sme sa vyhli opakujúcim sa chodcom, pretože dáta sú z videí a obávali sme sa preučenia 
    - Transformácia do grayscale 
    - Warp chodcov na 64x64 
 - Augmentácia dát 
    - Naklonovanie a rotácia obrázkov tých tried, ktoré sú nevyvážené, to nám ale prinášalo imbalance aj do iných tried nakoľko jeden chodec má až 4 triedy a každá trieda ma niekoľko labels 
 - Experimentovanie s hyperparametrami prostredníctvom WandB 
    - Šírka dense layers  
    - Batch size 
    - Počet epôch 
    - Learning rate 
    - Momentum pre SGD optimalizátor 
- Experimentovanie s architektúrou siete (VGG,ResNet) 
    - Pridanie reziduálnych blokov 
    - Pridanie, odstránenie skrytých vrstiev 
- Pridanie šírky dense vrstvy 
- Pridanie hĺbky 
    - Dense – 6 vrstiev 
    - CNN 64x64 -> Leaky Relu -> CNN 64x64 -> Leaky Relu -> Maxpool 2x 
- Dropout na úrovni 0.20 
- Pridanie early stopping 
- Rozbehanie na najlepších parametroch 
- Evaluácia 
    - Classification report 
    - Confusion matrix 
- Klasifikácia chodca do 4 tried – viac-výstupný klasifikátor - chodec chodí/stojí, pohlavie chodca, pozeranie chodca do kamery a prechádzanie chodca cez cestu 
- Binary Cross Entropy Loss function so Sigmoid layer pre binárne klasifikácie a Cross Entropy pre multi classification(cross trieda – crossing, not-crossing, crossing irrelevant) 
- Zverejnený GitHub public repozitár s dokumentáciou prostredníctvom README