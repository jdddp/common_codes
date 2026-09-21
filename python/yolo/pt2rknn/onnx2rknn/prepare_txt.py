import os

dirpath = './rknn_quantitation'
txtpath = './imagelist_cy.txt'
os.remove(txtpath)
f = open(txtpath, 'a+', encoding='utf-8')
for imgname in os.listdir(dirpath):
    f.write(os.path.join(dirpath,imgname)+'\n')
