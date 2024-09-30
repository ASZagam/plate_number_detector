#import the necessary libraries
from ultralytics import YOLO
import cv2
# import cvzone
import numpy as np
import streamlit as st


st.write("PLATE NUMBER DETECTION APP USING YOLO MODEL")
#load the fine tune model
model = YOLO("best (2).pt")
image_placeholder = st.empty()


#load the image for the object detection
upload_image = st.file_uploader("Choose image file...", type= ["jpg", "jpeg", "png"])
#imag = cv2.imread("runs/detect/train18/weights/test.jpg")




if upload_image:
    print("image successfully uploaded")
    file_byte = np.asarray(bytearray(upload_image.read()), dtype=np.int8)
    img = cv2.imdecode(file_byte, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    #convert the image to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    
    st.image(img, "Uploaded image", use_column_width=True)

    if st.button("Predict"):
        st.text("Image is being processed...")
        
        #insert the image to the model and detect object with confidence 0.25 and above
        results = model(img, conf = 0.50)
        
        #get the class name i.e plate_number
        class_name = model.names
        
        
        # iterate through the results to get information about the detected object
        for result in results:
            #select the bounding boxes from the information detected by the model
            boxes = result.boxes
            #loop through the bounding boxes
            for box in boxes:
                #select the coordinate of the box for the first object
                x1,y1, x2, y2 = box.xyxy[0]
                #convert the coordinate to integer
                x1,y1, x2, y2  =  int(x1), int(y1), int(x2), int(y2) 
                #draw the rectangle which include the image, top and buttom coordinate, color, tickness
                cv2.rectangle(img, (x1,y1),(x2,y2), (255,0,0), 5)
                #get the confidence for the corresponding object
                conf = box.conf[0]
                #get the class for the corresponding object also
                clas = box.cls[0]
                #put the tezt on the box I.e class and confidence, point to place the text, font, scale, color, and tickness
                cv2.putText(img, f"{class_name[int(clas)]} {conf:.2f}", (x1,y1),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
        st.image(img, "Detected image....", use_column_width=True)
        # cv2.destroyAllWindows()

    # while True:
    # #show the detected image
    #     cv2.imshow("Detected Image", img)   
    #     #press q to exit the image
    #     if cv2.waitKey(1) == ord("q"):
    #         break
        


