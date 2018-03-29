# -*- coding:utf-8 -*-
import string
import os
import shutil
import uuid
from captcha.image import ImageCaptcha
import numpy as np

import itertools

NUMBER_PER_IMAGE = 5
NUMBER_PER_PERMUTATION = 1
IMAGE_HEIGHT = 100
IMAGE_WIDTH = 40 + 20 * NUMBER_PER_IMAGE 
BASE_PATH = './datasets'
TEST_SET_SIZE_RATIO = 0.2

def create_captcha_image(image, subpath, captcha_set):
	create_path = os.path.join(BASE_PATH, subpath)
	if not os.path.exists(create_path):
		os.makedirs(create_path)

	captcha = ''.join(str(x) for x in captcha_set)
	fn = os.path.join(create_path, '%s_%s.png' % (captcha, uuid.uuid4()))
	image.write(captcha, fn)

def gen_captcha():
	if os.path.exists(BASE_PATH):
		shutil.rmtree(BASE_PATH)
	os.makedirs(BASE_PATH)
	image = ImageCaptcha(width=IMAGE_WIDTH, height=IMAGE_HEIGHT)

	generated_image_count = 0

	## training set
	for n in range(NUMBER_PER_PERMUTATION):
		for i in itertools.permutations(range(10), NUMBER_PER_IMAGE):
			create_captcha_image(image, 'train', i)
			generated_image_count += 1

	## test set
	for n in range(int(generated_image_count * TEST_SET_SIZE_RATIO)):
		random_set = np.random.randint(10, size=NUMBER_PER_IMAGE)
		create_captcha_image(image, 'test', random_set)



if __name__ == '__main__':
	gen_captcha()