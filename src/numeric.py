import numpy as np
class Numeric(object):
    def __init__(self):
        pass

    def compute_content_grad(self, F_content, F, layer):
        """
        :param F: the image to be generated layer activations
        :param F_content: the original content image's activations
        :param layer: which layer to compute
        :return: loss and grad
        """
        (Fl, Pl) = (F[layer], F_content[layer])
        El = Fl - Pl
        loss = sum(El ** 2) / 2
        grad = El * (Fl > 0)
        return loss, grad

    def compute_style_grad(self, G_style, G, F, layer):
        """
        :param G_style: the original image's G style activations
        :param G: the image to be generated style activations
        :param F: the image to be generated layer activations
        :param layer: which layer to compute
        :return:
        """
        (Al, Gl) = (G_style[layer], G[layer])
        Fl = F[layer]
        temp = Gl - Al
        constant = Al.shape[0]**-2 * Al.shape[1]**-2
        loss = 0.25 * constant * sum(temp**2)
        grad = constant * np.dot(Fl, temp) * (Fl > 0)
        return loss, grad

    def computeFandG(self, x, net, style_layers, content_layers):
        """
        :param x: the input to the caffe net layer
        :param net: the caffe.Net object used to do feature extraction
        :param style_layers: style_layers, for which we want to compute the style output
        :param content_layers: content_layers, for which to compute the activations
        :return: the computed activation and style response
        we should save all the activation for styles
        """
        if not x.shape == net.blobs['data'].shape[1:]:
            x = np.reshape(x, net.blobs['data'].shape[1:])
        net.blobs['data'].data[...] = x
        net.forward()
        F = {}
        G = {}
        all_layers = set(list(style_layers) + list(content_layers))
        for layer in all_layers:
            act = net.blobs[layer].data[0, ...].copy()
            act = np.reshape(act.shape[0], -1)
            F[layer] = act
            if layer in style_layers:
                G[layer] = np.dot(act, act)
        return F, G


    def computeLossAndGradAll(self, x, net, layers, F_content, G_style,
                              style_layers, content_layers, ratio=1e3):
        """
        :param x: the input to net
        :param net: the caffe.Net object
        :param F_content: precomputed F_content for content image activations
        :param layers: we want to keep the layers in order so we have to ask the caller to give the layers
        :param G_style: precomputed G_style for style image activations
        :param style_layers: style_layers to compute style
        :param content_layers: content_layers to compute content
        :param alpha: the contribute of content
        :param beta: the contribute of style
        :return: the total loss and grad
        """
        # first feed the x to net and get loss and grad
        styles = style_layers.keys()
        contents = content_layers.keys()
        (F, G) = self.computeFandG(x, net, styles, contents)
        # now do the backward layer by layer to compute grad and loss
        start_layer = layers[-1]
        net.blobs[start_layer].diff[:] = 0
        loss = 0
        for count in reversed(range(len(layers))):
            layer = layers[count]
            next_layer = None if count == 0 else layers[count-1]
            grad = net.blobs[layer].diff[0, ...]
            # the content loss and grad for this layer
            if layer in content_layers:
                weight = content_layers[layer]
                (tmpl, tmpg) = self.compute_content_grad(F_content, F, layer)
                loss += tmpl * weight
                grad += tmpg * weight

            # the style part
            if layer in style_layers:
                weight = style_layers[layer]
                (tmpl, tmpg) = self.compute_style_grad(G_style, G, F, layer)
                loss += tmpl * weight * ratio
                grad += tmpg * weight * ratio
            # flow to next
            net.blobs[layer].diff[0,...] = grad
            net.backward(start=layer, end=next_layer)
        grad = net.blobs['data'].diff[0, ...]
        grad = grad.flatten().astype(np.float64)
        return loss, grad



























