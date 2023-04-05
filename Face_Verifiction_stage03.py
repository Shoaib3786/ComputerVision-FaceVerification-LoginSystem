""" Goal: """
# Finally Face Verifiction step using Euclidean distance

""" Libraries """
import numpy as np

# Euclidean distance
def euclid_dist(db_user_embeddings, current_user_embeddings):

    euclidean_dist = np.linalg.norm(db_user_embeddings - current_user_embeddings)

    return euclidean_dist

# Cosine similarity calculation
def cosine_sim(db_user_embeddings, current_user_embeddings):

  a = np.matmul(np.transpose(db_user_embeddings), current_user_embeddings)

  b = np.sum(np.multiply(db_user_embeddings, current_user_embeddings))

  c = np.sum(np.multiply(db_user_embeddings, current_user_embeddings))

  cosine_similarity = 1 - (a / (np.sqrt(b) * np.sqrt(c)))

  return cosine_similarity


# face verification calculation
def verify_face(db_user_embeddings, current_user_embeddings):

    # setting a threshold value
    threshold_cosine = 0.45  # cosine
    threshold_euclid = 120  # Euclid

    # Euclidean Distance
    euclid_distance = euclid_dist(db_user_embeddings, current_user_embeddings)

    """to verify"""
    # if euclid_distance < threshold_euclid:
    #     print('By Euclidean - Face verified!!')
    #
    # else:
    #     print("By Euclidean - Face isn't verified!!")


    # Cosine Similarity
    cosine_similarity = cosine_sim(db_user_embeddings, current_user_embeddings)

    """to verify"""
    # if cosine_similarity < threshold_cosine:
    #     print('By Cosine similarity - Face verified!!')
    #
    # else:
    #     print("By Cosine similarity - Face isn't verified!!")

    return euclid_distance, cosine_similarity, threshold_cosine, threshold_euclid