import numpy as np

def im2col(img, h, w, stride=1, pad=0):
    """
    img: (B, C, H, W)
    h:   kernel height
    w:   kernel width
    """
    B, C, H, W = img.shape
    img = np.pad(img, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    out_h = (H + 2*pad - h)//stride + 1
    out_w = (W + 2*pad - w)//stride + 1

    out = np.zeros((B, C, h, w, out_h, out_w))

    for y in range(h):
        y_ = y + stride*out_h
        for x in range(w):
            x_ = x + stride*out_w
            out[:, :, y, x, :, :] = img[:, :, y:y_:stride, x:x_:stride]

    out = out.transpose(0, 4, 5, 1, 2, 3).reshape(B*out_h*out_w, -1)
    return out


class Conv2D:
    """
    in_ch:    number of input channels
    out_ch:   number of output channels
    h, w:     kernel size
    stride:   stride of conv process
    pad:      padding size
    img:      input image (B, in_ch, H, W)
    """
    def __init__(self, in_ch, out_ch, h, w, stride=1, pad=0):
        self.out_ch = out_ch
        self.stride = stride
        self.pad = pad
        self.filters = np.random.randn(out_ch, in_ch, h, w)
        self.bias = np.zeros(out_ch)

    def forward(self, img):
        B, in_ch, H, W = img.shape
        out_ch, in_ch, h, w = self.filters.shape

        img = im2col(img, h, w, self.stride, self.pad)
        filters = self.filters.reshape(out_ch, -1).T

        out = np.dot(img, filters) + self.bias

        out_h = 1 + int((H + 2*self.pad - h) / self.stride)
        out_w = 1 + int((W + 2*self.pad - w) / self.stride)
        out = out.reshape(B, out_h, out_w, -1).transpose(0, 3, 1, 2)
        return out


# edge detection
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from PIL import Image
    img = Image.open('./keboard.png').convert('L')
    img = np.array(img, dtype=float) / 255
    img = img[np.newaxis, np.newaxis, :, :]
    conv = Conv2D(in_ch=1, out_ch=1, h=3, w=3, stride=1, pad=1)
    ## horizontal edge detection
    conv.filters[0, 0, :, :] = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    ## vertical edge detection
    conv.filters[0, 0, :, :] = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    out = conv.forward(img)
    out = np.clip(out, 0, 1)
    
    # show the input and output image
    plt.subplot(121)
    plt.imshow(img[0, 0], cmap='gray')
    plt.title('Input image')
    plt.subplot(122)
    plt.imshow(out[0, 0], cmap='gray')
    plt.title('Output image')
    plt.savefig('./conved_output.png')
    plt.show()