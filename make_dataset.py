import os
from csv import reader
import csv
from skimage import io
import pandas as pd
from PIL import Image

rootdir = 'IMFDB_final'

seen2 =set()
for hero in os.listdir(rootdir):
    if not hero.startswith('.'):
        path1 = 'IMFDB_final/'+str(hero)
        print("----------------")
        print(hero)
        print("----------------")
        for movie in os.listdir(path1):
            if not movie.startswith('.'):
                print(movie)
                path2 = path1+ str("/")+ str(movie)+ str("/images")
                txt_path = path1 + str("/") + str(movie) + str("/") + str(movie) + str(".txt")
                txt_path_another = path1 + str("/") + str(movie) + str("/") + str(movie) + str("_another.txt")

                try:
                    seen = set()
                    with open(txt_path) as fin, open(txt_path_another, 'w') as fout:
                        for line in fin:
                            if not line.isspace() and not line in seen:
                                line1 = line.replace(' ', ',')
                                fout.write(line1.replace('\t', ','))
                                seen.add(line)


                    csv_path = path1 + str("/") + str(movie) + str("/") + str(movie) + str(".csv")
                    os.rename(txt_path_another, csv_path)

                    actual_csv = "IMFDB_dataset/data.csv"
                    with open(actual_csv,'a') as actualcsvfile:
                        csvwriter = csv.writer(actualcsvfile)

                        with open(csv_path,'r') as csvfile:
                            csv_reader = reader(csvfile)
                            for row in csv_reader:
                                if not row[1] in seen2:
                                    seen2.add(row[1])
                                    fields = [row[1], row[11]]
                                    # writing the fields
                                    csvwriter.writerow(fields)
                                    image_path = path2 + str("/")+ row[2]
                                    to_path = 'IMFDB_dataset/images/' + str(row[1])
                                    img = io.imread(image_path)
                                    io.imsave(to_path,img)
                except Exception as e:
                    print(e)
                    continue



df = pd.read_csv('IMFDB_dataset/data.csv')
df.to_csv('IMFDB_dataset/data.csv', index=False)


image_path = 'IMFDB_dataset/images/'
listing = os.listdir(image_path)
count = 0
for file in listing:
    count+=1
    print(count)
    im = Image.open(image_path + file)
    imsize=(256,256)
    if imsize[0] != -1 and im.size != imsize:
        if imsize[0] > im.size[0]:
            im = im.resize(imsize, Image.BICUBIC)
        else:
            im = im.resize(imsize, Image.ANTIALIAS)
    im.save(image_path + file)



"""
import csv
from csv import reader
seen2 =set()
count1 = 0
count2 = 0
with open('IMFDB_dataset/data.csv','r') as csvfile:
    csv_reader = reader(csvfile)
    for row in csv_reader:
        abc = row[0] + str(",") + row[1]
        if abc in seen2:
            count2+=1
            continue
        count1+=1
        seen2.add(abc)

print(count1)
print(count2)

image_path = 'IMFDB_dataset/images/'
listing = os.listdir(image_path)
with open('IMFDB_dataset/another.csv','a') as another:
    writer = csv.writer(another)
    with open('IMFDB_dataset/data.csv','r') as csvfile:
        csv_reader = reader(csvfile)
        for row in csv_reader:
            if row[4] in listing:
                writer.writerow(row)


image_path = 'IMFDB_dataset/images/'
listing = os.listdir(image_path)
with open('IMFDB_dataset/another.csv','r') as csvfile:
    csv_reader = reader(csvfile)
    for row in csv_reader:
        if row[0] not in listing:
            print(row[0])

"""