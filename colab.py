COLAB script


!pip install seeme
import seeme
from seeme import Client
cl = Client()
my_username = "darekdajcz"
my_password = "Uzi2115"

cl.login(my_username, my_password)
import fastai

application_id = cl.get_application_id(
    base_framework = "pytorch",
    framework = "fastai",
    base_framework_version = str(torch.__version__),
    framework_version = str(fastai.__version__),
    application = "image_classification"
)

model_name = "newML"
description = "..."

my_model = cl.create_model({
    "name": model_name,
    "description": description,
    "application_id": application_id,
    "auto_convert": True
})

cl.upload_model(my_model["id"], str(learn.path))