import roboflow

rf = roboflow.Roboflow(api_key="IxLNMP2xMrPwHszMHG4E")
model = rf.workspace().project("people-detection-o4rdr").version("11").model
prediction = model.download()