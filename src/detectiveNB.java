import java.io.*;
import java.lang.reflect.Array;
import java.util.*;

/** Due to me, finding out the requirement for a second algorithm, both Naive Bayes and KNN classifiers are written
 * within the same class. */

public class detectiveNB {
    private ArrayList<ArrayList<Integer>> trainData;
    private ArrayList<ArrayList<Integer>> testData;
    private int pp0;
    private int pp1;
    private int sample;
    private int numfeatures;
    private int [] nmcArr0;
    private int [] nmcArr1;
    private ArrayList<Integer> likelihood1given0;
    private ArrayList<Integer> likelihood0given0;
    private ArrayList<Integer> likelihood1given1;
    private ArrayList<Integer> likelihood0given1;

    /** Constructor for the class, uses some private methods to read the data and initializes some values */
    detectiveNB(String trainDataLoc, String testDataLoc){
        trainData = primeData(trainDataLoc);
        testData = primeData(testDataLoc);
        // Get Total Rows Of Training Data
        sample = getTotalDataRows(trainData);
        numfeatures = getTotalDataFeatures(trainData);
        pp0 = 0;
        pp1 = 0;
    }

    /** Small private class that helps maintaining the priority queue for the KNN algorithm  */

    private static class Pair {
        int i;
        double j;


         Pair (int x, double y){
            i = x;
            j = y;
        }

        int getX(){
            return i;
        }
        double getY(){
            return j;
        }
    }

    /** Reads the data from the .csv files, returns a 2D ArrayList */
    private ArrayList<ArrayList<Integer>> primeData(String fileLoc){
        ArrayList<ArrayList<Integer>> aList = new ArrayList<ArrayList<Integer>>();
        int i = 0;
        try{
            BufferedReader reader = new BufferedReader(new FileReader(fileLoc));
            for (String line = reader.readLine(); line != null; line = reader.readLine()){
                ArrayList<Integer> templist = new ArrayList<Integer>();
                aList.add(templist);
                String[] lineArr = line.split(",");
                for(String charstr : lineArr){
                  aList.get(i).add(Integer.parseInt(charstr));
                }
                i++;
            }
        }catch(FileNotFoundException fnfe){
            System.err.println("File could not be found in the given location");

        }catch(IOException ioe){
            System.err.println("Unexpected Format");
        }
        return aList;
    }

    /** Getter */
    int getTotalDataRows(ArrayList<ArrayList<Integer>> list){
        return list.size();
    }

    /** Getter *///Columns
    int getTotalDataFeatures(ArrayList<ArrayList<Integer>> list){
        return list.get(0).size()-1;
    }

    /**Given a value, this method calculates the prior probability of the training dataset */
    int getPriorProbability(int val){
        int count = 0;
        for (ArrayList<Integer> outer : trainData){
            if(outer.get(0).equals(val)){count++;}
        }
        return count;
    }

    /** Given a value(binary option), this method calculates the normalizing constant for each column in the dataset
     *
     * returns an array of constants based on the value*/
    int[] getNormalizingConstant(int val){
        int [] array = new int [numfeatures];
        int count;
        for(int i = 1; i< numfeatures+1; i++){
            count =0;
            for(int j = 0; j< sample;j++){
                if(trainData.get(j).get(i).equals(val)){
                    count++;
                }
            }
            array[i-1] = count;

        }
        return array;
    }

    /** When passed an outcome and a given , this method calculates the likelihood of an outcome given
     * the (binary) value.
     *
     * Returns a list of likelihoods */
    public ArrayList<Integer> calculateLikelihood(int outcome, int given){
        ArrayList<Integer> arrayList = new ArrayList<Integer>();
        int count;
        for(int i = 1; i< numfeatures+1; i++){
            count =0;
            for(int j = 0; j< sample;j++){
                if(trainData.get(j).get(0).equals(outcome) && trainData.get(j).get(i).equals(given)){
                    count++;
                }
            }
            arrayList.add(count);
        }
        return arrayList;
    }


    /** P(A|C) = P(C|A)*P(A)/P(C)
     *
     * gets results of the prior probability calculations based on the binary options
     * gets the arrays of normalizing constants
     * also orders the calculation of likelihoods */

    private void train(){
        // get prior probability values for 0 and 1 -->  P(A)
        pp0 = getPriorProbability(0);
        pp1 = getPriorProbability(1);

        //get normalization constant array --> P(C)
        nmcArr0 = getNormalizingConstant(0);
        nmcArr1 = getNormalizingConstant(1);

        //calculcate likelihood --> P(C|A)
        likelihood0given0 = calculateLikelihood(0,0);
        likelihood0given1 = calculateLikelihood(0,1);
        likelihood1given0 = calculateLikelihood(1,0);
        likelihood1given1 = calculateLikelihood(1,1);
    }

    /**Given an entry, calculates the probabilistic values of both outcomes and makes a decision between two
     *
     * returns a boolean indicating whether it is a correct guess or not*/
    private boolean calculateEntry(ArrayList<Integer> entry){
        int result = entry.get(0);
        double outcomeNormal = pp1/(double)sample;
        double outcomeAbnormal = pp0/(double)sample;
        double normalizingFactor = 1.0;

        for (int i = 1; i < numfeatures+1; i++){
            if(entry.get(i) == 0){
                outcomeNormal *=(likelihood1given0.get(i-1)/(double)pp1) + 0.00000000001;
                outcomeAbnormal *= (likelihood0given0.get(i-1)/(double)pp0) + 0.000000001;
                normalizingFactor*=(nmcArr0[i-1]/(double)sample)+ 0.000000000000001;
            }else{
                outcomeNormal *= (likelihood1given1.get(i-1)/(double)pp1)+0.000000000001;
                outcomeAbnormal *= (likelihood0given1.get(i-1)/(double)pp0)+0.000000000001;
                normalizingFactor *= nmcArr1[i-1]/(double)sample+0.00000000000001;
            }
        }

        double normalProb = outcomeNormal/ normalizingFactor;
        double abnormalProb = outcomeAbnormal/normalizingFactor;

        //System.out.println("normalprob: "+normalProb +"abnormal prob: "+ abnormalProb);
        if (result == 0){
            return abnormalProb > normalProb;
        }else{
            return normalProb > abnormalProb;
        }
    }


    /** Calculates the Euclidean distance between the given entry(person in the train data)
     * and all the points in the training data
     *
     * uses and returns a Priority Queue to avoid the need to sort data
     * */
    PriorityQueue<Pair> calculateEuclideanDistance(ArrayList<Integer> entry){
        PriorityQueue<Pair> allPoints = new PriorityQueue<Pair>(numfeatures - 1, new Comparator<Pair>() {
            @Override
            public int compare(Pair o1, Pair o2) {
                return Double.compare(o1.getY(), o2.getY());
            }
        });

        double presquareroot = 0.0;
        int temp;
        double sqrtoftemp;

        for(int i=0; i < sample; i++){
            for (int j = 1; j<numfeatures; j++){
                temp = entry.get(j) - trainData.get(i).get(j);

                presquareroot += temp*temp;
            }
            sqrtoftemp = Math.sqrt(presquareroot);
            presquareroot = 0.0;
            Pair tempPair = new Pair(i,sqrtoftemp);
            allPoints.add(tempPair);
        }
        return allPoints;
    }

    /** Calculates the Euclidean distance between the given entry(person in the train data)
     * and all the points in the training data
     *
     * uses and returns a Priority Queue to avoid the need to sort data
     * */

    PriorityQueue<Pair> calculateHammingDistance(ArrayList<Integer> entry){
        PriorityQueue<Pair> allPoints = new PriorityQueue<Pair>(numfeatures - 1, new Comparator<Pair>() {
            @Override
            public int compare(Pair o1, Pair o2) {
                return Double.compare(o1.getY(), o2.getY());
            }
        });

       int mismatch = 0;

        for(int i=0; i < sample; i++){
            for (int j = 1; j<numfeatures; j++){
                if(!entry.get(j).equals(trainData.get(i).get(j))){
                    mismatch++;
                }
            }
            Pair tempPair = new Pair(i,mismatch);
            allPoints.add(tempPair);
        }
        return allPoints;
    }

    /** Given an entry from the test data and a priority queue of distances (Hamming or Euclidean) and a value of k:
     *makes a decision based on the outcomes of the first k elements
     *
     * returns true if the guess based on first K element matches with the actual result
     * otherwise returns false*/


    boolean decideBasedOnKthNearestElements(ArrayList<Integer> entry, PriorityQueue<Pair> pQ, int k){
        int localK = 0;
        int near1 = 0;
        int near0 =0;
        while (localK < k){
            Pair pair = pQ.poll();
            if(trainData.get(pair.getX()).get(0) == 0){
                near0++;
            }else{
                near1++;
            }
            localK++;
        }


        if(near0>= near1){

            return entry.get(0) == 0;

        }else{
            return entry.get(0) == 1;
        }
    }

    /** Wrapper method for making a decision based on the Hamming distances, for each entry in the test dataset
     *
     *
     * Prints the result */
    void detectForKthHamm(int k){
        int count;
        int count0s= 0;
        int count1s =0;
        int total1s=0;
        int total0s= 0;


        for (ArrayList<Integer> entry : testData) {
            PriorityQueue<Pair> HamList = calculateEuclideanDistance(entry);

            if(entry.get(0).equals(0)){
                if(decideBasedOnKthNearestElements(entry, HamList, k)){
                    count0s++;
                }
                total0s++;
            }else{
                if(decideBasedOnKthNearestElements(entry, HamList, k)){
                    count1s++;
                }
                total1s++;
            }
        }

        count=count0s+count1s;
        System.out.print("total: " + (count)+"/"+testData.size() + "("+count/(double)testData.size()+")"+
                " abnormal: " + count0s +"/"+total0s+"("+count0s/(double)total0s +") normal: "+ count1s+
                "/"+total1s+"("+count1s/(double)total1s+")");
    }

    /** Wrapper method for making a decision based on the Hamming distances, for each entry in the test dataset
     *
     *
     * Prints the result */

    void detectForKthEuc(int k){
        int count;
        int count0s= 0;
        int count1s =0;
        int total1s=0;
        int total0s= 0;


        for (ArrayList<Integer> entry : testData) {
            PriorityQueue<Pair> EucList = calculateHammingDistance(entry);

            if(entry.get(0).equals(0)){
                if(decideBasedOnKthNearestElements(entry, EucList, k)){
                    count0s++;
                }
                total0s++;
            }else{
                if(decideBasedOnKthNearestElements(entry, EucList, k)){
                    count1s++;
                }
                total1s++;
            }
        }

        count=count0s+count1s;
        System.out.print("total: " + (count)+"/"+testData.size() + "("+count/(double)testData.size()+")"+
                " abnormal: " + count0s +"/"+total0s+"("+count0s/(double)total0s +") normal: "+ count1s+
                "/"+total1s+"("+count1s/(double)total1s+")");
    }

    /** Wrapper method for the detection based on Naive Bayesian. Goes through each element and counts the number
     *of correct decisions and prints the results. a*/

     void detect(){
        int count =0;
        int count0s= 0;
        int count1s =0;
        int total1s=0;
        int total0s= 0;

        for (ArrayList<Integer> entry : testData){

            if (entry.get(0).equals(0)){
                if(calculateEntry(entry)){
                    count0s++;
                }
                total0s++;
            } else{
                if(calculateEntry(entry)){
                    count1s++;
                }
                total1s++;
            }
        }
        count=count0s+count1s;
        System.out.print("total: " + (count)+"/"+testData.size() + "("+count/(double)testData.size()+")"+
                " abnormal: " + count0s +"/"+total0s+"("+count0s/(double)total0s +") normal: "+ count1s+
                "/"+total1s+"("+count1s/(double)total1s+")");
    }

    /** Helper method for debugging*/
    public void displayTrainData(){
        displayList(trainData);
    }

    /** Helper Method for debugging*/
    public void displayTestData(){
        displayList(testData);
    }

    /** Helpor method for debugging */
    public ArrayList<ArrayList<Integer>> getTrainData(){
        return trainData;
    }

    /** Helper method for debugging */

    public ArrayList<ArrayList<Integer>> getTestData(){
        return testData;
    }

    /** Helper method for debugging*/
    private void displayList(ArrayList<ArrayList<Integer>> list){
        for (ArrayList<Integer> outer : list){
            for(Integer val : outer){
                System.out.print(val +" ");
            }
            System.out.println();
        }
    }

    public static void main (String [] args){
        detectiveNB detective = new detectiveNB(args[0],args[1]);

        if(args.length == 2){
            System.out.println();
            detective.train();
            detective.detect();
        }else{
            System.out.println("Knth Neareast Algorithm:");
            detective.detectForKthHamm(Integer.parseInt(args[2]));
        }
        System.out.println();
    }
}
