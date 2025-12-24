import logging
logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(levelname)s - %(message)s")

class ModelTraining:
    def train_model(model,X_train,y_train):
        model.fit(X_train,y_train)
        train_score = model.score(X_train,y_train)
        logging.info("Model trained succesfully")
        return model,train_score
