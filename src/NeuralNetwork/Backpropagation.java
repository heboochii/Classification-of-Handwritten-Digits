package NeuralNetwork;

//this class contains the code for backpropagation
//backpropagation is used to calculate the change in output neurons and hidden layer neurons
//it then updates the weights for output and hidden layer neurons

public class Backpropagation extends Functions{

    //this function initiates backpropagation by calculating change in output neurons
    public static void calculateOutputLayerDeltas(double digit) {
        double target;
        for (int i = 0; i < OLayerOutput.length;i++) {
            if(i == digit) {
                target=1;
            }
            else {
                target=0;
            }
            dOutputNeurons[i]= OLayerOutput[i]*(1-OLayerOutput[i])*(OLayerOutput[i]-target);
        }
    }

    //this function continues with backpropagation by calculating change in hidden layer neurons
    public static void calculateHiddenLayerDeltas() {
        for (int i = 0; i < hiddenLayerNos; i++) {
            dHiddenNeurons[i] =0;
            for (int k =   0; k <  outputLayerNos; k++ ) {
                dHiddenNeurons[i] +=	((dOutputNeurons[k] * (outputLayerWeights[k][i])));
            }
            dHiddenNeurons[i]*=(1 - HLayerOutput[i]) * HLayerOutput[i];
        }
    }

    //this function updates the weights for output and hidden layer neurons(backpropagation)
    //the threshold is set 1
    public static void updatingOandHLWeights() {
        for (int i = 0; i < outputLayerNos; i++) {
            for (int k = 0; k < nosOfOLayerWeights - 1; k++ ) { outputLayerWeights[i][k] += -learningRate *
                    dOutputNeurons[i] * HLayerOutput[k];
            }
            //threshold is 1
            outputLayerWeights[i][nosOfOLayerWeights-1] += -learningRate * dOutputNeurons[i];
        }
    }

    //this function updates the weights in the hidden layer (backpropagation)
    public static void updatingHLWeights(double[] input) {
        for (int i = 0; i < hiddenLayerNos; i++) {
            for (int k = 0; k <  nosOfHLayerWeights - 1; k++ ) {
                hiddenLayerWeights[i][k] += -learningRate *  dHiddenNeurons[i] * input[k];
            }
            //new weight - updated weight
            hiddenLayerWeights[i][nosOfHLayerWeights-1] += -learningRate *  dHiddenNeurons[i];
        }
    }
}
