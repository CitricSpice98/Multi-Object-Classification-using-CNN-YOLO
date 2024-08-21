from ultralytics import YOLO
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
# model.train(data="coco128.yaml", epochs=3)  # train the model
# metrics = model.val()  # evaluate model performance on the validation set
#print(os.getcwd())
cardataset =os.getcwd() + "/dataset/train/cars/"
bikesdataset =os.getcwd() + "/dataset/train/bikes/"
persondataset =os.getcwd() + "/dataset/train/person/"
res_car = []
res_bikes = []
res_person = []
person_files = os.listdir(persondataset)
car_files = os.listdir(cardataset)
bike_files = os.listdir(bikesdataset)

y_true = []
y_pred = []

for img in car_files:
    # print('img is ', img)
    results = model(cardataset + img)  # predict on an image
    #print(results[0].boxes)
    #print(results[0].probs)
    predicted = results[0].boxes.cls.numpy()
    #print('predicted is ', predicted)
    result_label = []
    y_true.append(0)
    for key in predicted:
        result_label.append(results[0].names.get(key,-1))
    #print('result_label is ', result_label)
    if "car" in result_label:
        res_car.append(1)
        y_pred.append(0)
    else:
        res_car.append(0)
        y_pred.append(3)

res_car_correct = 0
for i in range(len(res_car)):
    if res_car[i] == 1:
        res_car_correct += 1


for img in bike_files:
    # print('img is ', img)
    results = model(bikesdataset + img)  # predict on an image
    #print(results[0].boxes)
    #print(results[0].probs)
    predicted = results[0].boxes.cls.numpy()
    #print('predicted is ', predicted)
    result_label = []
    y_true.append(1)
    for key in predicted:
        result_label.append(results[0].names.get(key,-1))
    #print('result_label is ', result_label)
    if "bicycle" in result_label:
        res_bikes.append(1)
        y_pred.append(1)
    else:
        res_bikes.append(0)
        y_pred.append(3)

res_bike_correct = 0
for i in range(len(res_bikes)):
    if res_bikes[i] == 1:
        res_bike_correct += 1


for img in person_files:
    # print('img is ', img)
    results = model(persondataset + img)  # predict on an image
    #print(results[0].boxes)
    #print(results[0].probs)
    predicted = results[0].boxes.cls.numpy()
    #print('predicted is ', predicted)
    result_label = []
    y_true.append(2)
    for key in predicted:
        result_label.append(results[0].names.get(key,-1))
    #print('result_label is ', result_label)
    if "person" in result_label:
        res_person.append(1)
        y_pred.append(2)
    else:
        res_person.append(0)
        y_pred.append(3)

res_person_correct = 0
for i in range(len(res_person)):
    if res_person[i] == 1:
        res_person_correct += 1

print('len of res car is ', len(res_car))

print('len of res bikes is ', len(res_bikes))

print('len of res person is ', len(res_person))

print('accuracy on car dataset is ', res_car_correct/len(res_car)*100)

print('accuracy on bikes dataset is ', res_bike_correct/len(res_bikes)*100)

print('accuracy on person dataset is ', res_person_correct/len(res_person)*100)

# Create confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Car", "Bikes", "Person", "Unrecognized"],
            yticklabels=["Car", "Bikes", "Person", "Unrecognized"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig("confusion_matrix_yolo_model.png")
plt.show()
