from ultralytics import YOLO
model = YOLO("yolo26s.pt")          # downloads the weights the first time
result = model.predict(r"C:\Users\oremo\Desktop\baka-naman-may-bakante\data\raw\cepdof\Edge_cases\Edge_cases_000069.jpg", classes=[0], conf=0.25)[0]
print("People found:", len(result.boxes))
result.save("results/test_prediction.jpg")  # open this file to see the boxes