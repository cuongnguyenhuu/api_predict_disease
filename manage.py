#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
# from Predict import views



def main():
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'api_predict_disease.settings')
    # views.load_model()
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)
    

# def init_model():
#     # CREATE THE MODEL ARCHITECTURE
#     mobile = tf.keras.applications.mobilenet.MobileNet()
#     # Exclude the last 5 layers of the above model.
#     # This will include all layers up to and including global_average_pooling2d_1
#     x = mobile.layers[-6].output

#     # Create a new dense layer for predictions
#     # 7 corresponds to the number of classes
#     x = Dropout(0.25)(x)
#     predictions = Dense(7, activation='softmax')(x)

#     # inputs=mobile.input selects the input layer, outputs=predictions refers to the
#     # dense layer we created above.

#     model = Model(inputs=mobile.input, outputs=predictions)
#     model.load_weights("./Predict/model.h5")
#     print("AAA")



if __name__ == '__main__':
    main()
    