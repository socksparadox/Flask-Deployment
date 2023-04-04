# Flask-Deployment
In this project I've made a simple Machine Learning project on Wine Quality Classification.
We have dataset which has 12 features namely pH level, citric acid, free sulfur dioxide, etc.
After the process of Data wrangling we try different machine learning models for the data.
We compare the accuracies of various models from the Results table and choose the model with best accuracy and save it to pickle file.
Now that we have our model, we will import Flask library and load our model there from the pickle file.
We will use a index.html file which will be our template as to how many feautres we will require and how we will get the output.
We then write the code to get output from the model after providing inputs.
Also a file is attached named style.css which is in the static folder. This is where you can make changes in the sizes of the objects on the webpage.
To deploy the model, run the app.py file. A link will be generated. Paste the link in a web browser to get the webpage.
