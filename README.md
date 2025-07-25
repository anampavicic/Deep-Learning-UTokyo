# Deep-Learning-UTokyo
This is a repository for the Deep learning for Perception: Recognition, classification and generation class at University of Tokyo

contributor: Bitzan Michael, Pavičić Ana Marija, Staeblein Lorena


### Data
You can look at the raw data files under the directory *data*. Preprocessing ran through the *data_augmentation.ipynb* and has also been saved under *data*. The *ClassesData* directory entails the *AnimalSoundDataset.py*, where we read the data out and formatted it into an usable dataset.

### Model
All the classes used for the model itself can be found in *ClassesML* with a similar structure to the project in class. You can also find the definition of the training in *ClassesML/AudioTrainer.py*

### Optimization
We ran hyperparameter tuning via the file *hyperparameter_tuning.py* and saved the first results under *results*. Under *configs/audio_model_modified_data_best.json* and *audio_model_processed_data_best*  you can read in the best hyperparameter from the tuning.

After this the 5-fold Cross-Validation can be found under *main.py*, where we also run the model.

### Results
The results are depicted via an accuracy-history graphic and a conclusion matrix, both of which can be found in *main.py*.