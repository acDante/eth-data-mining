from random import randint

# number of bands and rows inside a band of the partitioned signature matrix
# calculated for a little less than 85% similarity to reduce false negatives
b = 47
r = 21
# number of hash functions used for min-hashing
min_hash_size = r*b
# prime numbers used by the hash functions 
prime1 = 1073741827
prime2 = 1073741843
# the output spaces of the hash functions
buckets1 = 8193
buckets2 = 10000000
# a, b uniformly random integers used for min-hashing
# they remain the same throughout different calls of the mapper
a = [randint(1, buckets1) for i in range(min_hash_size)]
b = [randint(0, buckets1) for i in range(min_hash_size)]
# c, d uniformly random integers used for hashing bands
# they remain the same throughout different calls of the mapper
c = [randint(1, buckets2) for i in range(r)]
d = [randint(0, buckets2) for i in range(r)]

def mapper(key, value):
    # key: None
    # value: one line of input file, a video

    # extract the indexes of the singles of a video
    shingles = map(int, value.split()[1:])

    # calculate the signature matrix for this video
    signature = []
    for i in range(min_hash_size): 
        temp = []
        for z in shingles:
            temp.append(((a[i]*z + b[i])%prime1)%buckets1)
        signature.append(min(temp))
     
    # split the signature matrix into b bands of r rows
    bands = [signature[i:i + r] for i in range(0, len(signature), r)]    
    
    # hash the bands of the signature matrix for this video
    band_hashes = []
    for band in bands: 
        temp = []
        for j in range(r):
            temp.append(((c[j]*band[j] + d[j])%prime2)%buckets2)
        band_hashes.append(sum(temp)%buckets2)
    
    # emit key, value pairs with
    # key: a band's id concatenated with the band's hash
    # value: the video
    for i in range(len(band_hashes)):
        key = str(i) + "," + str(band_hashes[i])
        yield key, value


# calculate the Jaccard similarity between two lists
def similarity(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    return len(set1 & set2)*1.0/len(set1 | set2)


def reducer(key, values):
    # key: a band's id concatenated with the band's hash
    # values: list of videos having the same band hashed to the same bucket
     
    # sort the list of videos 
    values.sort()

    # extract id and shingles from each video
    video_ids = []
    video_shingles = []
    for value in values:
        # extract the video id
        video_id = int(value.split()[0].split("_")[1])
        # extract the indexes of the singles of a video
        shingles = map(int, value.split()[1:])
        video_ids.append(video_id)
        video_shingles.append(shingles)

    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            # for each candidadte pair of videos calculate the similarity
            if similarity(video_shingles[i], video_shingles[j]) >= 0.85:
                yield video_ids[i], video_ids[j]

