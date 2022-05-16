from typing import Dict
import zipfile
import os
import shutil

import boto3


def s3_cp_unzip(s3_zip_path: str, output_dir_path: str, extra_arg: Dict = None) -> str:
    """AWS S3 cp and unzip for zip file
    ...

    >>> s3_zip_path = 's3://id-ai-function/id_object_attribute/v_0_13_0_id_object_attribute_object_feature_tflite.zip'
    >>> extra_arg = {'VersionId': 'e4mfJtLlbJEQd0v48s0IU3r_xA53a_0P'}
    >>> output_dir_path = s3_cp_unzip(s3_zip_path, '/tmp/s3_cp_unzip', extra_arg)
    '/tmp/s3_cp_unzip'

    Args:
      s3_zip_path: S3uri path
      output_dir_path: Output directory path
      extra_arg: Extra arguments that may be passed to the client operation (optional).

    Returns:
        Output directory path

    Raises:
        ClientError: An error occurred (403 or 404) when calling the HeadObject operation: (Forbidden or Not Found).
        Please check AWS secret key or s3_zip_path.
    """

    try:
        os.makedirs(output_dir_path, exist_ok=True)
        s3 = boto3.resource('s3')
        bucket_name = s3_zip_path.split('/')[2]
        key_name = '/'.join(s3_zip_path.split('/')[3:])
        bucket = s3.Bucket(bucket_name)
        output_zip_path = os.path.join(output_dir_path, os.path.basename(key_name))
        if not os.path.exists(output_zip_path):
            bucket.download_file(key_name, output_zip_path, ExtraArgs=extra_arg)

            with zipfile.ZipFile(output_zip_path) as existing_zip:
                existing_zip.extractall(output_dir_path)
            os.remove(output_zip_path)
    except:
        shutil.rmtree(output_dir_path, ignore_errors=True)
        raise
    return output_dir_path
