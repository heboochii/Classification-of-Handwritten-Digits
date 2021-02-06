package NeuralNetwork;

//In this class, all the function, algorithm are called
//It contains the main method

public class Main extends Functions{
    public static void main(String[] args) throws Exception{
        //Run test loop
        testLoop = 0;
        while (testLoop <= 1) {
            //Executes the below functions
            readDS(testLoop);
            trainingPerceptron();
            testingPerceptron();
            testLoop++;
        }
    }
}
