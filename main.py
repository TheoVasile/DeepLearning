### main python file used for testing neural network
import DeepLearning as dl
from DeepLearning import Nets
import pygame as pg

#initialize pygame module
pg.init()

img_width = 28
img_height = 28

#initialize screen variables
width = 500
height = 500
screen = pg.display.set_mode((width, height))
pg.display.set_caption("Neural Net")
clock = pg.time.Clock()
fps = 10

image = pg.image.load("pine-chris-image.jpg")
image = pg.transform.scale(image, (img_width, img_height))
pixels = pg.surfarray.array3d(image)
inputs = []
for c in pixels:
    for p in c:
        inputs.append(p[0]/255)
        inputs.append(p[1]/255)
        inputs.append(p[2]/255)
#print(inputs)

autoencoder = Nets.NeuralNetwork([img_height*img_width, 100, img_height*img_width], 1)

#start gameloop
running = True
while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

    autoencoder.feedForward(inputs)
    autoencoder.backPropagate(inputs)
    output = autoencoder.get_layer(-1)

    screen.fill((255, 255, 255))

    index = 0
    #print(output)
    print(len(output))
    for y in range(0, img_height-1):
        for x in range(0, img_width-1):
            try:
                pg.draw.rect(screen, (int(output[index] * 255), int(output[index+1] * 255), int(output[index+2]) * 255), (int(x)*4, int(y)*4, 4, 4), 0)
            except:
                print(index)
            index += 3
    pg.display.update()
    clock.tick(fps)