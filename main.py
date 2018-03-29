# -*- coding:utf-8 -*-
import datetime
import sys 
import os
import numpy as np
import tensorflow as tf
from PIL import Image

MAX_STEPS = 10000
BATCH_SIZE = 50
TEST_SIZE = 2000

LOG_DIR = 'log/captcha-%s' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
BASE_PATH = './datasets'
MODEL_DIR = './model'
MODEL = os.path.join(MODEL_DIR, 'cnn-captcha.ckpt')

NUMBER_PER_IMAGE = 5
NUMBER_PER_PERMUTATION = 1
IMAGE_HEIGHT = 100
IMAGE_WIDTH = 40 + 20 * NUMBER_PER_IMAGE 
LABEL_CHOICES = "0123456789"
LABEL_LENGTH = len(LABEL_CHOICES)


def read_image(filename):
    ## Grayscale로 변환하고, 이미지를 균일한 사이즈로 맞춘다.
    image = Image.open(filename).convert('L').resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.ANTIALIAS)
    ## 데이터 어레이의 참조를 생성한다.
    data = np.asarray(image)
    return data

def read_label(filename):
    ## 파일 이름을 받아온다.
    basename = os.path.basename(filename)
    ## 파일 이름을 쪼개서 캡챠의 답 (label)을 가져온다.
    labels = basename.split('_')[0]

    ## label의 데이터 어레이 
    data = []

    ## 캡챠 label(해답)의 각 숫자를 인덱스로 바꾼다.
    for char in labels:
        ## 인덱스 찾기 
        index = LABEL_CHOICES.index(char)
        ## 인덱스의 위치를 표시하기 위해 10개의 0으로 채워진 어레이를 생성
        tmp = [0] * LABEL_LENGTH
        ## 해당 인덱스를 1로 표시한다.
        tmp[index] = 1
        ## 데이터 추가
        data.extend(tmp)

    return data

def read_images_and_labels(dir_name):
    ## png 확장자만 읽는다.
    ext = '.png'
    ## 이미지 데이터
    images = []
    ## 레이블 데이터 
    labels = []
    ## 해당 경로의 파일리스트 순회
    for file_name in os.listdir(dir_name):
        ## png 확장자만 선택한다.
        if file_name.endswith(ext):
            ## 파일 경로를 얻어온다.
            file_path = os.path.join(dir_name, file_name)
            ## 각 이미지 어레이와 레이블 어레이를 가져온다.
            images.append(read_image(file_path))
            labels.append(read_label(file_path))
    ## 데이터의 복사본을 리턴한다.
    return np.array(images), np.array(labels)

def load_data():
    ## 트레인 셋 경로
    train_dir = os.path.join(BASE_PATH, 'train')
    ## 테스트 셋 경로 
    test_dir = os.path.join(BASE_PATH, 'test')
    ## 트레인 이미지 셋과 레이블 셋
    train_images, train_labels = read_images_and_labels(train_dir)
    ## 테스트 이미지 셋과 레이블 셋
    test_images, test_labels = read_images_and_labels(test_dir)
    ## 튜플로 리턴 
    return (
        (train_labels, train_images),
        (test_labels, test_images)
    )

## 해당 데이터 셋의 다음 배치 데이터 셋을 리턴하는 함수 
def get_next_batch(data, batch_size, last_offset):
    labels = data[0]
    images = data[1]

    ## 행렬 shape의 첫번째 차원은 전체 이미지 개수
    data_length = images.shape[0]

    ## 다음 오프셋이 데이터 크기를 초과한다면, 0으로 초기화
    if last_offset + batch_size > data_length:
        last_offset = 0

    ## 오프셋이 0이면 무작위로 기존 리스트를 셔플하여 Step에 따라 데이터 셋이 골고루 선택되도록 한다.
    if last_offset == 0:
        ## 데이터 크기만큼 0~length까지의 순열 어레이 생성
        shuffle_indexes = np.arange(data_length)
        ## 셔플
        np.random.shuffle(shuffle_indexes)
        ## 기존 데이터의 각 위치를 셔플된 인덱스 위치로 재정렬한다.
        images = images[shuffle_indexes]
        labels = labels[shuffle_indexes]
    ## 데이터의 시작 오프셋
    start = last_offset
    ## 데이터의 최종 오프셋
    last_offset += batch_size
    return images[start:last_offset], labels[start:last_offset], last_offset


def main(argv):
    ## 데이터 셋을 가져온다.
    train_data, test_data = load_data()
    print('[Data Loaded] Training Set: %s. Test Set: %s' % 
        (train_data[1].shape[0], test_data[1].shape[0]))

    ## Input
    x = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH], name='x')
    y = tf.placeholder(tf.float32, [None, NUMBER_PER_IMAGE * LABEL_LENGTH], name='y')
    x_image = tf.reshape(x, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
    tf.summary.image('input', x_image, max_outputs=LABEL_LENGTH)

    ## Convolution 1
    W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))

    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1,1,1,1], padding='SAME') + b_conv1)
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    ## Convolution 2
    W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))

    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1,1,1,1], padding='SAME') + b_conv2)
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    ## Fully connected
    W_fc1 = tf.Variable(tf.truncated_normal([IMAGE_WIDTH * IMAGE_HEIGHT * 4, 1024], stddev=0.1))
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))

    h_pool2_flat = tf.reshape(h_pool2, [-1, IMAGE_WIDTH * IMAGE_HEIGHT * 4])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    ## Dropout
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    ## Readout
    W_fc2 = tf.Variable(tf.truncated_normal([1024, NUMBER_PER_IMAGE * LABEL_LENGTH], stddev=0.1))
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[NUMBER_PER_IMAGE * LABEL_LENGTH]))

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    ## Reshape
    y_expect_reshaped = tf.reshape(y, [-1, NUMBER_PER_IMAGE, LABEL_LENGTH])
    y_got_reshaped = tf.reshape(y_conv, [-1, NUMBER_PER_IMAGE, LABEL_LENGTH])

    ## Loss
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_expect_reshaped, logits=y_got_reshaped))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    ## forward feed
    predict = tf.argmax(y_got_reshaped, axis=2, name='predict')
    expect = tf.argmax(y_expect_reshaped, axis=2)

    ## evaluate 
    correct_prediction = tf.equal(predict, expect)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    ## 모든 변수의 저장과 복구를 위한 오퍼레이션 추가
    saver = tf.train.Saver()

    ## 세션 시작
    with tf.Session() as sess:
        ## summary operation을 머지
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(LOG_DIR + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(LOG_DIR + '/test', sess.graph)

        ## 변수 초기화 
        tf.global_variables_initializer().run()

        ## 최근 읽어들인 데이터 오프셋 
        last_offset_train = 0
        last_offset_test = 0

        ## 반복 학습 
        for i in range(MAX_STEPS):
            ## 한 번에 읽어들일 배치 데이터 가져오기 
            train_x, train_y, last_offset_train = get_next_batch(train_data, BATCH_SIZE, last_offset_train)

            ## 해당 배치 서머지와 러닝, 각 플레이스홀더로 데이터셋과 드롭아웃되지 않을 확률을 전달
            step_summary, _ = sess.run([merged, train_step], feed_dict={x: train_x, y: train_y, keep_prob: 1.0})
            train_writer.add_summary(step_summary, i)

            ## 100 번에 한 번 씩 정확도를 계산한다.
            if i % 100 == 0:
                ## 트레이닝 셋의 정확도를 계산한다.
                train_summary, train_accuracy = sess.run([merged, accuracy], feed_dict={x: train_x, y: train_y, keep_prob: 1.0})
                train_writer.add_summary(train_summary, i)

                ## 테스트 셋 데이터를 가져온다.
                test_x, test_y, last_offset_test = get_next_batch(test_data, TEST_SIZE, last_offset_test)

                ## 테스트 셋의 정확도를 계산한다.
                test_summary, test_accuracy = sess.run([merged, accuracy], feed_dict={x: test_x, y: test_y, keep_prob: 1.0})
                test_writer.add_summary(test_summary, i)

                ## 데이터를 출력한다.
                print('[step: %s] training accuracy: %.2f%%, test accuracy: %.2f%%' % (i, train_accuracy * 100, test_accuracy * 100))

        train_writer.close()
        test_writer.close()

        ## 최종 데이터 평가를 위한 테스트 셋을 가져온다.
        test_x, test_y, last_offset_test = get_next_batch(test_data, TEST_SIZE, last_offset_test)

        ## 테스트 정확도 계산
        test_accuracy = accuracy.eval(feed_dict={x: test_x, y: test_y, keep_prob: 1.0})
        print('Final test ccuracy: %.2f%%' % (test_accuracy * 100))

        ## 최종 학습된 모델을 파일로 저장한다.
        save_path = saver.save(sess, MODEL)
        print('Model saved in file: %s' % save_path)

if __name__ == '__main__':
    tf.app.run(main=main)
