import java.lang.reflect.Array;
import java.util.Arrays;


public class Network {

    /*LAYER DETAILS*/
    public final int[] NETWORK_LAYER_SIZES;  //Number of neurons in each layer
    public final int INPUT_SIZE;             //Size of input
    public final int OUTPUT_SIZE;            //Size of output
    public final int NETWORK_SIZE;           //Total number of layers in the network

    /*DATA DETAILS OF EACH LAYER*/
    private double[][] bias;       //First array is Layer, Second array is neuron, This is because there is a single bias connected to every neuron
    private double[][] output;     //First array is Layer, Second array is neuron
    private double[][][] weights;  /*First array is Layer, Second array is neuron, Third array is previous neuron.
                                     Previous neuron is taken into account as every neuron has multiple weights connected to it,
                                     to identify the correct weight we need to know the previous neuron to which it is connected*/

    private double[][] error_signal;
    private double[][] output_derivative;


    /*CONSTRUCTOR*/
    public Network(int... NETWORK_LAYER_SIZES){

        /*LAYER DETAILS*/
        this.NETWORK_LAYER_SIZES = NETWORK_LAYER_SIZES;
        this.INPUT_SIZE = NETWORK_LAYER_SIZES[0];               //First layer of the network is the input
        this.NETWORK_SIZE = NETWORK_LAYER_SIZES.length;         //Total number of layers in the network = length of NETWORK_LAYER_SIZES array
        this.OUTPUT_SIZE = NETWORK_LAYER_SIZES[NETWORK_SIZE-1]; //Last layer of Network is output, it is given by (total layers - 1)

        this.output = new double[NETWORK_SIZE][];
        this.weights = new double[NETWORK_SIZE][][];
        this.bias = new double[NETWORK_SIZE][];

        this.error_signal = new double[NETWORK_SIZE][];
        this.output_derivative = new double[NETWORK_SIZE][];

        for (int i = 0; i < NETWORK_SIZE; i++){
            this.output[i] = new double[NETWORK_LAYER_SIZES[i]];
            this.error_signal[i] = new double[NETWORK_LAYER_SIZES[i]];
            this.output_derivative[i] = new double[NETWORK_LAYER_SIZES[i]];

            //assigned random value between given ranges to bias
            this.bias[i] = NetworkTools.createRandomArray(NETWORK_LAYER_SIZES[i],0.3,0.7);

            if(i > 0){
                //assigned random value between given ranges to weights
                this.weights[i] = NetworkTools.createRandomArray(NETWORK_LAYER_SIZES[i],NETWORK_LAYER_SIZES[i-1],-0.5,0.5);
            }
        }
    }

    public double[] calculate(double... input){
        if(input.length != this.INPUT_SIZE) return null;
        this.output[0] = input;

        for(int layer = 1; layer < NETWORK_SIZE; layer++){
            for(int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++){
                double sum = bias[layer][neuron];
                for(int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer-1]; prevNeuron++){
                    sum += output[layer-1][prevNeuron] * weights[layer][neuron][prevNeuron];
                }
                output[layer][neuron] = sigmoid(sum);
                output_derivative[layer][neuron] = (output[layer][neuron] * (1 - output[layer][neuron]));
            }
        }
        return output[NETWORK_SIZE-1];
    }

    public double sigmoid (double x){
        return ((2d / (1 + Math.exp(-x)))-1);
    }

    public void backpropError(double[] target){
        for(int neuron = 0; neuron < NETWORK_LAYER_SIZES[NETWORK_SIZE-1]; neuron++){
            error_signal[NETWORK_SIZE-1][neuron] = (target[neuron] - output[NETWORK_SIZE-1][neuron])
                    * output_derivative[NETWORK_SIZE-1][neuron];
            System.out.println(error_signal[NETWORK_SIZE-1][neuron]);
        }
        for(int layer = NETWORK_SIZE-2; layer > 0; layer--){
            for(int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++){
                double sum = 0;
                for(int nextNeuron = 0; nextNeuron < NETWORK_LAYER_SIZES[layer+1]; nextNeuron++){
                    sum += weights[layer + 1][nextNeuron][neuron] * error_signal[layer + 1][nextNeuron];
                }
                this.error_signal[layer][neuron] = sum * output_derivative[layer][neuron];
            }
        }
    }

    public void updateWeights(double eta){
        for (int layer = 1; layer < NETWORK_SIZE; layer++){
            for(int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron ++){
                for(int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer-1]; prevNeuron++){
                    double delta = eta * output[layer-1][prevNeuron] * error_signal[layer][neuron];
                    weights[layer][neuron][prevNeuron] += delta;
                }
                double delta = eta * error_signal[layer][neuron];
                bias[layer][neuron] += delta;
            }
        }
    }

    public void train(double[] input, double[] target, double eta){
        //System.out.println("Entered train");
        if(input.length != INPUT_SIZE || target.length != OUTPUT_SIZE) return;
        calculate(input);      //calculate the output
        backpropError(target);
        updateWeights(eta);
    }

    public static void main(String[] args){
        Network net = new Network(2,4,1);

        double[] input_1 = new double[]{-1,-1};
        double[] target_1 = new double[]{-1};

        double[] input_2 = new double[]{-1,1};
        double[] target_2 = new double[]{1};

        double[] input_3 = new double[]{1,-1};
        double[] target_3 = new double[]{1};

        double[] input_4 = new double[]{1,1};
        double[] target_4 = new double[]{-1};

        for (int i = 0; i < 3500;i++){
            net.train(input_1,target_1,0.2);
            net.train(input_2,target_2,0.2);
            net.train(input_3,target_3,0.2);
            net.train(input_4,target_4,0.2);
        }


        double[] o = net.calculate(input_2);

        System.out.println("Output"+Arrays.toString(o));
    }
}
