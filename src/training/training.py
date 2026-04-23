# Train the model

print("Starting model training...")
history = model.fit(
    X_train, Y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, Y_val),
    callbacks=[early_stopping],
    verbose=1
)

print("Training completed!")



# Save the trained model
model_filename = 'trained_model.h5'
model.save(model_filename)
print(f"Model saved as: {model_filename}")


# Save training history
history_filename = 'model_history.json'
with open(history_filename, 'w') as f:
    json.dump(history.history, f)
print(f"Training history saved as: {history_filename}")