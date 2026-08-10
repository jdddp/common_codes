import os

dirpath = './cy'
txtpath = './imagelist_cy.txt'
f = open(txtpath, 'a+', encoding='utf-8')
for imgname in os.listdir(dirpath):
    f.write(os.path.join(dirpath,imgname)+'\n')
