import base64

#with open("encoded-image.txt", "rb") as text_file:
#    encoded_image = text_file.read()
#    print(encoded_image)


with open("0002.jpg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read())
print(encoded_string)

#data=b"merhaba"
#kodlanan_veri=base64.b64encode(data)
#print(kodlanan_veri)

#with open("0002.jpg", "rb") as f:
#    data = f.read()

#print(data)
#print(base64.b64encode(data))

#transmitted_data=base64.b64decode(data)
#print(transmitted_data)