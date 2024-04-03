from aimodelshare.aimsonnx import model_to_onnx

# Save tf.keras model (or any tensorflow model) to local ONNX file
def save_model_onxx(model, model_filename, framework='keras', transfer_learning=False, deep_learning=False):
    onnx_model = model_to_onnx(
        model,
        framework=framework,
        transfer_learning=transfer_learning,
        deep_learning=deep_learning
    )

    with open(model_filename, "wb") as f:
        f.write(onnx_model.SerializeToString())
        print(f"Model saved to {model_filename}")

def upload_preds_to_aimodelshare(model, model_filename, X_test, y_test, experiment, preprocessor_filename='preprocessor.zip'):
    # Generate predictions and extract the index of the highest probability
    prediction_column_index = model.predict(X_test).argmax(axis=1)
    # Extract the text labels for the highest probability index
    prediction_labels = [y_test.columns[i] for i in prediction_column_index]
    # Submit predictions to competition Leaderboard
    experiment.submit_model(
        model=model_filename,
        preprocessor=preprocessor_filename,
        prediction_submission=prediction_labels
    )