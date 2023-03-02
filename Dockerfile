FROM public.ecr.aws/lambda/python:3.8
RUN yum install -y mesa-libGLw
# Copy function code
COPY app.py ${LAMBDA_TASK_ROOT}
COPY face_geometry.py ${LAMBDA_TASK_ROOT}
COPY keypoint_classifier.py ${LAMBDA_TASK_ROOT}
COPY keypoint_classifier.tflite ${LAMBDA_TASK_ROOT}
# RUN ls -laR ${LAMBDA_TASK_ROOT}/*

# Install the function's dependencies using file requirements.txt
# # from your project folder.
# RUN yum install -y gcc-c++
RUN yum install -y gcc

RUN yum -y install make
COPY Makefile .
COPY pthread_shim.c .
RUN make pthread_shim.so && cp pthread_shim.so /opt
COPY requirements.txt  .
# RUN yum install ffmpeg
RUN pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "app.lambda_handler" ]