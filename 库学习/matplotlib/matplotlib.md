```python
for i,picture_dir in enumerate(train_normal_picture+train_penumonia_picture):
        sp = plt.subplot(4,4,i+1)    #多个位置的哪一个框
        sp.axis('off')               #不要标注刻度
        if i < 8:
            sp.set_title("NORMAl")   #设置标题
        else:
            sp.set_title('PENUMONIA')
        img = mpimg.imread(picture_dir)    # 读取照片
        plt.imshow(img)                    # 照片的展示
plt.show()
```

