import io
from PIL import Image
import configparser

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, FloatType


def get_model_for_eval():
    """Gets the broadcasted model."""
    model = models.resnet50(pretrained=True)
    model.load_state_dict(bc_model_state.value)
    model.eval()
    return model


def pil_loader(binary_file):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    image_io = io.BytesIO(binary_file)
    img = Image.open(image_io)
    return img.convert('RGB')


# Create a custom PyTorch dataset class.
class ImageDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = pil_loader(self.paths[index])
        if self.transform is not None:
            image = self.transform(image)
        return image


# Define the function for model inference.
# PyArrow >= 1.0.0 must be installed;
@pandas_udf(ArrayType(FloatType()))
def predict_batch_udf(paths: pd.Series) -> pd.Series:
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    images = ImageDataset(paths, transform=transform)
    loader = torch.utils.data.DataLoader(images, batch_size=500, num_workers=8)
    model = get_model_for_eval()
    model.to(device)
    all_predictions = []
    with torch.no_grad():
        for batch in loader:
            predictions = list(model(batch.to(device)).cpu().numpy())
            for prediction in predictions:
                all_predictions.append(prediction)
    return pd.Series(all_predictions)


# 这个配置文件，会在启动任务时集群自动配置
config = configparser.ConfigParser()
config.read("/opt/spark/default-config/spark-defaults.conf")

try:
    # 在集群中
    AWS_ACCESS_KEY_ID = config['aws']['spark.hadoop.fs.s3a.access.key']
    AWS_SECRET_ACCESS_KEY = config['aws']['spark.hadoop.fs.s3a.secret.key']
    AWS_ENDPOINT_ADDRESS = config['aws']['spark.hadoop.fs.s3a.endpoint']
    AWS_IMPL = config['aws']['spark.hadoop.fs.s3.impl']
    print("Starting Spark in cluster...")
except:
    # 在本地环境中
    AWS_ACCESS_KEY_ID = None
    AWS_SECRET_ACCESS_KEY = None
    AWS_ENDPOINT_ADDRESS = None
    AWS_IMPL = None
    print("Starting Spark in local...")

# 初始化连接
spark = SparkSession \
    .builder \
    .appName('torch-infer') \
    .config("spark.hadoop.fs.s3a.access.key", AWS_ACCESS_KEY_ID) \
    .config("spark.hadoop.fs.s3a.secret.key", AWS_SECRET_ACCESS_KEY) \
    .config("spark.hadoop.fs.s3a.endpoint", AWS_ENDPOINT_ADDRESS) \
    .config("spark.hadoop.fs.s3.impl", AWS_IMPL) \
    .getOrCreate()
sc = spark.sparkContext

cuda = True
# Enable Arrow support.
spark.conf.set("spark.sql.execution.arrow.enabled", "true")
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "64")

use_cuda = cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Load ResNet50 on driver node and broadcast its state.
model_state = models.resnet50(pretrained=True).state_dict()
bc_model_state = sc.broadcast(model_state)

# datasource: hdds/s3...
images = sc. \
    binaryFiles("hdfs://xxx/flower_photos/*/*") \
    .repartition(10)
df = spark.createDataFrame(data=images, schema=['filename', 'data'])

df.printSchema()
df.show(1)
# print(df.rdd.take(1))

# Make predictions.
# datasource: hdds/s3...
predictions_df = df.select(col("filename"), predict_batch_udf(col("data")).alias("prediction"))
predictions_df \
    .write \
    .mode("overwrite") \
    .parquet("hdfs://xxx/output/")

spark.stop()


