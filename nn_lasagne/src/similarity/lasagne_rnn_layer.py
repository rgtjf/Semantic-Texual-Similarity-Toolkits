import theano.tensor as T
import lasagne


def stack_rnn(l_emb, l_mask, num_layers, num_units,
              grad_clipping=10, dropout_rate=0.,
              bidir=True,
              name='',
              rnn_layer=lasagne.layers.LSTMLayer):
    """
        Stack multiple RNN layers.
    """

    def _rnn(backwards=True, name=''):
        network = l_emb
        for layer in range(num_layers):
            if dropout_rate > 0:
                network = lasagne.layers.DropoutLayer(network, p=dropout_rate)
            network = rnn_layer(network, num_units,
                                grad_clipping=grad_clipping,
                                mask_input=l_mask,
                                backwards=backwards,
                                name=name + '_layer' + str(layer + 1))
        return network

    network = _rnn(True, name)
    if bidir:
        network = lasagne.layers.ConcatLayer([network, _rnn(False, name + '_back')], axis=-1)
    return network
