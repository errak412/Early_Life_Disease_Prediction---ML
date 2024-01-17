import os

from sklearn import tree #Scikit-Learn, also known as sklearn is a python library to implement machine learning models and statistical modelling.
import numpy as np # NumPy can be used to perform a wide variety of mathematical operations on arrays.
import pandas as pd #Pandas is a Python library used for working with data sets. It has functions for analyzing, cleaning, exploring, and manipulating data
from matplotlib import pyplot as plt #Matplotlib is a popular data visualization library in Python. It's often used for creating static, interactive, and animated visualizations in Python. Matplotlib allows you to generate plots, histograms, bar charts, scatter plots, etc., with just a few lines of code.
import numpy as np
from flask import Flask, session, url_for, redirect, render_template, request, abort, flash #Flask is a small and lightweight Python web framework that provides useful tools and features that make creating web applications in Python easier.
from database import db_connect,view_p1,find_dis,add_bill,view_bill,payment_act,ps1,add_amount,status1,pmed1,pvmed,pmed_act,med_act,accp,docc,book_act,viewbilltable,upload_file
from database import patient_login,admin_login, doctor_login,patient_reg,doctor_reg,add_symptoms,view_patient
from datetime import date
import cv2 #OpenCV is a Python library that allows you to perform image processing and computer vision tasks. It provides a wide range of features, including object detection, face recognition, and tracking
from stegano import lsb #Stegano, a pure Python Steganography module. Steganography is the art and science of writing hidden messages in such a way that no one, apart from the sender and intended recipient, suspects the existence of the message, a form of security through obscurity.
# from chatterbot import ChatBot
# from chatterbot.trainers import ChatterBotCorpusTrainer
# from chatterbot.trainers import ListTrainer

from chatterbot import ChatBot #ChatterBot is a Python library that makes it easy to generate automated responses to a user's input. ChatterBot uses a selection of machine learning algorithms to produce different types of responses. This makes it easy for developers to create chat bots and automate conversations with users.
from chatterbot.trainers import ListTrainer
from chatterbot.trainers import ChatterBotCorpusTrainer

app = Flask(__name__)
app.secret_key = os.urandom(24)

@app.route("/") #App routing is the technique used to map the specific URL with the associated function intended to perform some task. The Latest Web frameworks use the routing technique to help users remember application URLs. It is helpful to access the desired page directly without navigating from the home page.
def FUN_root():
    return render_template("index.html")

@app.route("/admin.html")
def admin():
    return render_template("admin.html") 

@app.route("/home.html")
def home1():
    return render_template("home.html")

@app.route("/med.html")
def med():
    return render_template("med.html")

@app.route("/adminviewbill.html")
def adminviewbill():
    data1=viewbilltable()
    return render_template("adminviewbill.html",data=data1)

@app.route("/ps.html")
def ps():
    data=ps1()
    return render_template("ps.html",data=data)

@app.route("/pmed.html")
def pmed():
    data1=pmed1()
    return render_template("pmed.html",data=data1)

@app.route("/patientviewpres.html")
def patientviewpres():
    data1=view_bill()
    return render_template("patientviewpres.html",data=data1)     

@app.route("/addprescription.html")
def addprescription():
    data=view_patient()
    return render_template("addprescription.html", prescription=data)     

@app.route("/patient.html")
def patient():
    return render_template("patient.html")

@app.route("/viewPdetails.html")
def viewPdetails():
    data1=view_patient()
    return render_template("viewPdetails.html",data1=data1) 


@app.route("/viewpatientsymptom.html")
def viewpatientsymptom():
   
    return render_template("viewpatientsymptom.html") 

@app.route("/doctorhome.html")
def doctorhome():
    return render_template("doctorhome.html") 
    
@app.route("/adminhome.html")
def adminhome():
    return render_template("adminhome.html") 

@app.route("/patienthome.html")
def patienthome():
    return render_template("patienthome.html")     

@app.route("/doctor.html")
def doctor():
    return render_template("doctor.html") 
    
@app.route("/payment.html")
def payment():
    return render_template("payment.html") 

@app.route("/patientreg.html")
def patientreg():
    return render_template("patientreg.html") 

@app.route("/doctorreg.html")
def doctorreg():
    return render_template("doctorreg.html") 
 

@app.route("/payment1")
def payment1():
      id=request.args.get('Pid')
      data1=payment_act(id) 
      return render_template("payment.html",m1="sucess", data=data1)      

@app.route("/doc")
def doc1():
      data1=docc() 
      return render_template("doc.html",m1="sucess", data=data1) 

@app.route("/status")
def status():
      data1=status1() 
      return render_template("status.html",m1="sucess", data=data1)      
      
@app.route("/vmed.html")
def vmed():
      data1=pvmed() 
      return render_template("vmed.html",m1="sucess", data=data1) 

@app.route("/patientlogin", methods=['GET', 'POST'])
def patientlogin():
    if request.method == 'POST':
        status = patient_login(request.form['id'], request.form['psw'])
        print(status)
       
        if status == 1:
            return render_template("patienthome.html", m1="sucess",)
        else:
            return render_template("patient.html", m1="Login Failed") 

@app.route("/doctorlogin", methods=['GET', 'POST'])
def doctorlogin():
    if request.method == 'POST':
        status = doctor_login(request.form['id'], request.form['psw'])
        print(status)
        if status == 1:
          
            return render_template("doctorhome.html",  m1="sucess")
        else:
            return render_template("doctor.html", m2="Failed") 

@app.route("/adminlogin", methods=['GET', 'POST'])
def adminlogin():
    if request.method == 'POST':
        status = admin_login(request.form['name'], request.form['psw'])
        print(status)

        if status == 1:
            return render_template("adminhome.html",m1="sucess")
        else:
            return render_template("admin.html", m2="Failed") 
             

@app.route("/patientregact", methods = ['GET','POST'])
def patientregact():
   if request.method == 'POST':      
      status = patient_reg(request.form['id'],request.form['name'],request.form['email'],request.form['phone'],request.form['address'],request.form['psw'])
      if status == 1:
       return render_template("patient.html",m1="sucess")
      else:
       return render_template("patientreg.html",m1="failed")    

@app.route("/doctorregact", methods = ['GET','POST'])
def doctorregact():
   if request.method == 'POST':      
      status = doctor_reg(request.form['id'],request.form['name'],request.form['phone'],request.form['email'],request.form['address'],request.form['psw'])
      if status == 1:
       return render_template("doctor.html",m1="sucess")
      else:
       return render_template("doctorreg.html",m1="failed")   


@app.route("/bookact", methods = ['GET','POST'])
def bookact1():
   if request.method == 'POST':      
      status = book_act(request.form['id'],request.form['name'],request.form['fee'],request.form['dt'],request.form['pid'],request.form['pname'],request.form['cn'],request.form['cvv'],request.form['tt'])
      if status == 1:
       return render_template("book.html",m1="sucess")
      else:
       return render_template("book.html",m1="failed")  

@app.route("/pmedact", methods = ['GET','POST'])
def pmedact():
   if request.method == 'POST':      
      status = pmed_act(request.form['mname'],request.form['price'],request.form['f'],request.form['name'],request.form['id'],request.form['phone'],request.form['cn'],request.form['cvv'])
      if status == 1:
       return render_template("vmed.html",m1="sucess")
      else:
       return render_template("vmed.html",m1="failed")  

@app.route("/medact", methods = ['GET','POST'])
def medact1():
   if request.method == 'POST':      
      status = med_act(request.form['tname'],request.form['price'],request.form['image'])
      if status == 1:
       return render_template("med.html",m1="sucess")
      else:
       return render_template("med.html",m2="failed") 

# @app.route("/addsymptoms", methods = ['GET','POST'])
# def addsymptoms():
#    if request.method == 'POST':      
#       status = add_symptoms(request.form['id'],request.form['s1'],request.form['s2'],request.form['s3'],request.form['s4'],request.form['s5'])
#       if status == 1:
#        data1=session['id'] 
#        return render_template("patienthome.html",m1="sucess", data=data1)
#       else:
#        return render_template("addPsymptoms.html",m1="failed") 


@app.route("/addbill", methods = ['GET','POST'])
def addbill():
   if request.method == 'POST':      
      status = add_bill(request.form['id'],request.form['pres'],request.form['sugges'],request.form['amount'])
      if status == 1:
       return render_template("viewPdetails.html",m1="sucess")
      else:
       return render_template("addprescription.html",m1="failed")       

@app.route("/pidsub", methods = ['GET','POST'])
def pidsub():
   if request.method == 'POST':      
       t=request.form['id']
       print(t)
       data1=view_p1(t)
       return render_template("viewpatientsymptom.html", data=data1)

@app.route("/pay", methods = ['GET','POST'])
def pay():
   if request.method == 'POST':      
      status = add_amount(request.form['id'],request.form['price'],request.form['tno'],request.form['cvv'])
      if status == 1:
       return render_template("patientviewpres.html",m1="sucess")
      else:
       return render_template("patientviewpres.html",m1="failed")
    
@app.route("/Upload", methods = ['GET','POST'])
def owner_upload():
   if request.method == 'POST':
      file = request.files['inputfile']
      check = upload_file(request.form['fname'],file,session['username'],"No","No")
      if check == True:
         return render_template("fileupload.html",m1="success")
      else:
         return render_template("fileupload.html",m1="Failed")


@app.route("/addPsymptoms.html")
def addPsymptoms():
    l1=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
'yellow_crust_ooze']
   
    return render_template("addPsymptoms.html", data=l1) 

disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
' Migraine','Cervical spondylosis',
'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
'Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
'Impetigo']

# @app.route("/find")
# def find():
#       a=request.args.get('symptom1')
#       b=request.args.get('symptom2')
#       c=request.args.get('symptom3')
#       data1=find_dis(a,b,c) 
#       return render_template("viewresult.html",m1="sucess", data=data1)

@app.route("/find")      
def find():
    
    l1=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
'yellow_crust_ooze']
    l2=[]
    for x in range(0,len(l1)):
      l2.append(0)


    df=pd.read_csv("Training.csv") 
    df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)
    
    X= df[l1]
    y = df[["prognosis"]]
    np.ravel(y)
# print(y)

# TRAINING DATA tr --------------------------------------------------------------------------------
    tr=pd.read_csv("Testing.csv")
    tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

# # entry variables
#     Symptom1 = StringVar()
#     Symptom1.set(None)
#     Symptom2 = StringVar()
#     Symptom2.set(None)
#     Symptom3 = StringVar()
#     Symptom3.set(None)
#     Symptom4 = StringVar()
#     Symptom4.set(None)
#     Symptom5 = StringVar()
#     Symptom5.set(None)
#     Name = StringVar()

#     # from sklearn import tree

    clf3 = tree.DecisionTreeClassifier()   # empty model of the decision tree
    clf3 = clf3.fit(X,y)

    # # calculating accuracy-------------------------------------------------------------------
    # from sklearn.metrics import accuracy_score
    # y_pred=clf3.predict(X_test)
    # print(accuracy_score(y_test, y_pred))
    # print(accuracy_score(y_test, y_pred,normalize=False))
    # -----------------------------------------------------

    psymptoms = [request.args.get('symptom1'),request.args.get('symptom2'),request.args.get('symptom3'),request.args.get('symptom4'),request.args.get('symptom5')]
    for k in range(0,len(l1)):
        # print (k,)
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = clf3.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break


    if (h=='yes'):
        des=[]
        print(disease[a])
        # t1.delete("1.0", END)
        # t1.insert(END, disease[a])
        des=disease[a]
    else:
        # t1.delete("1.0", END)
        # t1.insert(END, "Not Found")
         print("not found")

    
    
    return render_template("viewresult.html",m1="sucess", data=des)

@app.route("/addsymptoms", methods = ['GET','POST'])
def addsymptoms():
    
    l1=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
'yellow_crust_ooze']
    l2=[]
    for x in range(0,len(l1)):
      l2.append(0)


    df=pd.read_csv("Training.csv") 
    df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)
    
    X= df[l1]
    y = df[["prognosis"]]
    np.ravel(y)
# print(y)

# TRAINING DATA tr --------------------------------------------------------------------------------
    tr=pd.read_csv("Testing.csv")
    tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

# # entry variables
#     Symptom1 = StringVar()
#     Symptom1.set(None)
#     Symptom2 = StringVar()
#     Symptom2.set(None)
#     Symptom3 = StringVar()
#     Symptom3.set(None)
#     Symptom4 = StringVar()
#     Symptom4.set(None)
#     Symptom5 = StringVar()
#     Symptom5.set(None)
#     Name = StringVar()

# 
# 


#     # from sklearn import tree

    clf3 = tree.DecisionTreeClassifier()   # empty model of the decision tree
    clf3 = clf3.fit(X,y)

    # # calculating accuracy-------------------------------------------------------------------
    # from sklearn.metrics import accuracy_score
    # y_pred=clf3.predict(X_test)
    # print(accuracy_score(y_test, y_pred))
    # print(accuracy_score(y_test, y_pred,normalize=False))
    # -----------------------------------------------------

    psymptoms = [request.form['id'],request.form['s1'],request.form['s2'],request.form['s3'],request.form['s4'],request.form['s5']]
    for k in range(0,len(l1)):
        # print (k,)
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = clf3.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break


    if (h=='yes'):
        des=[]
        print(disease[a])
        # t1.delete("1.0", END)
        # t1.insert(END, disease[a])
        des=disease[a]
    else:
        # t1.delete("1.0", END)
        # t1.insert(END, "Not Found")
         print("not found")

    
    
    if request.method == 'POST':      
      status = add_symptoms(request.form['id'],request.form['s1'],request.form['s2'],request.form['s3'],request.form['s4'],request.form['s5'],des)
      if status == 1:
       data1=session['id'] 
       return render_template("patienthome.html",m1="sucess", data=data1,des = des)
      else:
       return render_template("addPsymptoms.html",m1="failed")


@app.route("/book")
def book():
      a=request.args.get('a')
      b=request.args.get('b')
      e=request.args.get('e')
      return render_template("book.html",m1="sucess", a=a , b=b ,e=e)

@app.route("/mbook")
def mbook():
      a=request.args.get('a')
      b=request.args.get('b')
      c=request.args.get('c')
      return render_template("mbook.html",m1="sucess", a=a , b=b )

@app.route("/mbook2", methods = ['GET','POST'])
def mbook2():
      a=request.form['med']
      b=request.form['price']
      b1 = int(b)
      c=request.form['ns']
      c1 = int(c)
      f = b1 * c1
      return render_template("mbook2.html",m1="sucess", a=a , b=b , c=c, f=f )

@app.route("/acc1")
def acc1():
      a=request.args.get('a')
      e=request.args.get('e')
      f=request.args.get('f')
      accp(a,e,f) 
      return render_template("viewPdetails.html",m1="sucess")




# with open('file.txt','r') as file:
#     conversation = file.read()
# small_talk = ['hi there!',
#               'hi!',

#               'how do you do?',
#               'how are you?',

#               'i\'m cool.',
#               'fine, you?',
#               'always cool.',
#               'i\'m ok',
#               'glad to hear that.',
#               'i\'m fine',
#               'glad to hear that.',
#               'i feel awesome',
#               'excellent, glad to hear that.',
#               'not so good',
#               'sorry to hear that.',
#               'what\'s your name?',
#               'chotu',
#               'fever'
#               'i\'m pybot. ask me a math question, please.']

# bot = ChatBot("Hospital ChatBot")
# trainer = ListTrainer(bot)
# for item in (small_talk):
#     print("0000000000000")
#     print(item)
#     trainer.train(item)
    


# # @app.route("/")
# # def home():
# # 	return render_template("home.html")

# @app.route("/get")
# def get_bot_response():
# 	userText = request.args.get('msg')
# 	print("//////////////////////")
# 	print(str(bot.get_response(userText)))
# 	return str(bot.get_response(userText))


chatbot = ChatBot(
    'CHAT BOT',
    storage_adapter='chatterbot.storage.SQLStorageAdapter',
    logic_adapters=[
        'chatterbot.logic.BestMatch',
        {
        'import_path': 'chatterbot.logic.BestMatch',
        'default_response': 'I am sorry, but I do not understand. I am still learning.',
        'maximum_similarity_threshold': 0.90
        }
    ],
    database_uri='sqlite:///database.sqlite3'
) 
 # Training with Personal Ques & Ans 
training_data_quesans = open('C:/Users/Sai Shankar Katta/Desktop/hospital management chatbot/HEALTH.txt').read().splitlines()
#training_data_personal = open('training_data/simple.txt').read().splitlines()
#training_data_conv = open('training_data/more.txt').read().splitlines()

training_data = training_data_quesans
print(training_data)
trainer = ListTrainer(chatbot)
trainer.train(training_data) 
# Training with English Corpus Data 
trainer_corpus = ChatterBotCorpusTrainer(chatbot)


app.static_folder = 'static'

    
# @app.route("/")
# def home():
#     return render_template("index.html")
    
@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    print("dddddddddddddddddddddddddddd")
    #print(userText)
    return str(chatbot.get_response(userText))



# ----------------------------------------------Update Item------------------------------------------ 

# ----------------------------------------------Update Item------------------------------------------       
if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
