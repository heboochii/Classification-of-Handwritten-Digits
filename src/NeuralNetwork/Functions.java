package NeuralNetwork;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;

//this class contains the variable and arrays
//it contains the sigmoid activation function
//it contains a method that generates random values in a range for the weights
//it contains the buffered reader that reads the files

public class Functions {
    public static double[][] trainingDS = new double[2810][65]; //array for the training dataset
    public static double[][] testingDS = new double[2810][65]; //array for the testing dataset

    public static double result = 0; //variable for the result calculation
    public static double result1 = 0; //variable for the calculation result of training
    public static double result2 = 0; //variable for the calculation result of testing
    public static double avgResult = 0; //variable for the calculation result of average
    public static int hiddenLayerNos=60; //variable the hidden layers
    public static int outputLayerNos=10; //variable output

    public static double[] HLayerOutput = new double[hiddenLayerNos]; //array for the output in hidden layer
    public static double[] OLayerOutput = new double[outputLayerNos]; //array for output layer output
    public static double[] dHiddenNeurons = new double[hiddenLayerNos]; //array for change in (delta) hidden neurons
    public static double[] dOutputNeurons = new double[outputLayerNos]; //array for change in (delta) output neurons

    public static int rows=64; //variable for the input of the rows
    //public static int columnsInput = 2810; //variable for the input of the columns
    public static int nosOfHLayerWeights = rows + 1; //variable for hidden layer weights
    public static int nosOfOLayerWeights = hiddenLayerNos + 1; //variable for output layer weights

    public static double[][] hiddenLayerWeights = new double[hiddenLayerNos][nosOfHLayerWeights]; //array for hidden weight
    public static double[][] outputLayerWeights = new double[outputLayerNos][nosOfOLayerWeights]; //array for hidden weight
    public static int[][] desiredOutput = new int[10][10]; //array for key for desired output (10*10)

    public static double learningRate = 0.09; //variable for the learning rate
    public static int testLoop = 0;


    //this function activates the neurons and runs the sigmoid function using the equation
    private static double sigmoidFunction(double learningRateInput) {
        double activationFunction = 1 + Math.exp ( -learningRateInput );
        return 1/activationFunction;
    }


    //this is the part of the sigmoid function. in this, it calculates and gives the hidden layer output
    private static void getHLayerOutput(double[] image) {
        int i = 0;
        while (i < hiddenLayerNos) {
            double summationHiddenLayer = 0;
            for (int k = 0; k <  nosOfHLayerWeights - 1; k++ ){
                summationHiddenLayer +=image[k]*hiddenLayerWeights[i][k];
            }
            summationHiddenLayer += hiddenLayerWeights[i][nosOfHLayerWeights -1];
            HLayerOutput[ i++ ] = sigmoidFunction(summationHiddenLayer);
        }
    }

    //this is the part of the sigmoid function. in this, it calculates and gives the output layer output
    private static void getOLayerOutput() {
        int i = 0;
        while (i < outputLayerNos) {
            double summationOutputLayer = 0;
            for (int k =   0; k < nosOfOLayerWeights - 1; k++ ) {
                summationOutputLayer += HLayerOutput[k]*outputLayerWeights[i][k];
            }
            summationOutputLayer += outputLayerWeights[i][nosOfOLayerWeights - 1];
            OLayerOutput[i++] = sigmoidFunction(summationOutputLayer);
        }
    }

    //This function generates random weights for the training cycle
    private static void generateRandomWeights() {
        //range for the randomized generated weights
        double maxWeight = 1;
        double minWeight = -1;

        for (int hiddenLayer = 0; hiddenLayer < hiddenLayerNos; hiddenLayer++) {
            int inputLayer = 0;
            while (inputLayer < nosOfHLayerWeights) {
                hiddenLayerWeights[hiddenLayer][inputLayer]= minWeight +(Math.random()*(maxWeight - minWeight));
                inputLayer++;
            }
        }
        for ( int OutputLayer=0;  OutputLayer<outputLayerNos; OutputLayer++) {
            int outputWeight=0;
            while (outputWeight<nosOfOLayerWeights) {
                outputLayerWeights[OutputLayer][outputWeight]= minWeight +(Math.random()*(maxWeight - minWeight));
                outputWeight++;
            }
        }
    }

    //this function trains the weights for the perceptron by using randomly generated weights
    public static void trainingPerceptron() {
        generateRandomWeights();
        for (int i = 0; i < 60; i++){
            int value=0;
            while (value<2810) {
                getHLayerOutput(trainingDS[value]);
                getOLayerOutput();
                Backpropagation.calculateOutputLayerDeltas(trainingDS[value][64]);
                Backpropagation.calculateHiddenLayerDeltas();
                Backpropagation.updatingOandHLWeights();
                Backpropagation.updatingHLWeights(trainingDS[value]);
                value++;
            }
        }
    }

    //in this, it is the testing of the perceptron. it has the neuron output and gets the outputs from the datasets
    //and it also generates the output, i.e. accuracy in percents.
    public static void testingPerceptron() {
        int neuronOutput = 0;
        for (int i =0; i < 2810; i++) {
            getHLayerOutput(testingDS[i]);
            getHLayerOutput(testingDS[i]);
            getOLayerOutput();
            int k =   0;
            while (k <  outputLayerNos) {
                if (OLayerOutput[k] > 0.5){
                    neuronOutput=k;
                }
                k++;
            }
            if (neuronOutput == testingDS[ i ][ 64 ]) {
                result++; // its for both test. This contain the number of correct classification for each test.
            }
            //desired output
            desiredOutput[neuronOutput][(int) testingDS[i][64]]++;
        }

        // Generates output
        avgResult += result;

        if(testLoop==0) {
            result1 += result;
            result=0;
        }
        else {
            result2 += result;
            System.out.println("Accuracy for training data set: " + result1 * 100 / 2810 + "%");
            System.out.println("\nAccuracy for testing data set: " + result2 * 100 / 2810 + "%");
            System.out.println("\nAverage: "+ avgResult * 100 / 5620 + "%");

        }
    }


    //this functions calls the datasets and runs addingtoData
    public static void readDS(int testNum) throws Exception{
        String trainingDSPath, testingDSPath;
        if (testNum == 0) {
            trainingDSPath = "cw2DataSet2.csv";
            testingDSPath = "cw2DataSet1.csv";
        } else {
            trainingDSPath = "cw2DataSet1.csv";
            testingDSPath = "cw2DataSet2.csv";
        }

        addingToData(trainingDSPath, trainingDS);
        addingToData(testingDSPath, testingDS);

    }

    //this function gets the data lines using buffered reader and adds to the data line
    private static void addingToData(String fileName, double[][] trainData) throws Exception{
        int lineData = 0;
        try {
            BufferedReader reader = new BufferedReader(new FileReader(fileName));
            String line;
            while ((line = reader.readLine()) != null) {
                String[] data = line.split(",");
                for (int i = 0; i < 65; i++) {
                    trainData[lineData][i] = Integer.parseInt(data[i]);
                }
                lineData++;
            }
            reader.close();
        }
        //Throws Exception If File Not Found
        catch (FileNotFoundException ex) {
            ex.printStackTrace();
        }
    }

}