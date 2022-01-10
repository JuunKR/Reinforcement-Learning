import tensorflow as tf

tf.compat.v1.disable_eager_execution()

#@ 밴딧
# 밴딧의 손잡이 목록을 작성
# 현재 손잡이 4(인덱스는 3)가 가장 자주 양의 보상을 제공하도록 설정되어 있다.
bandit_arms = [0.2, 0, -0.2, -2]
num_arms = len(bandit_arms)
def pullBandit(bandit):
    # 랜덤한 값을 구한다.
    result = np.random.randn(1)
    if result > bandit:
        # 양의 보상을 반환한다.
        return 1
    else:
        # 음의 보상을 반환한다.
        return -1

#@ Agent
sess = tf.compat.v1.Session()

weights = tf.compat.v1.Variable(tf.ones([num_arms]))
output = tf.compat.v1.nn.softmax(weights)

init = tf.compat.v1.global_variables_initializer()

sess.run(init)
print(sess.run(weights))
print(sess.run(output))