import java.io.*;
import java.lang.reflect.Array;
import java.util.ArrayList;

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

    public detectiveNB(String trainDataLoc, String testDataLoc){
        trainData = primeData(trainDataLoc);
        testData = primeData(testDataLoc);
        // Get Total Rows Of Training Data
        sample = getTotalDataRows(trainData);
        numfeatures = getTotalDataFeatures(trainData);
        pp0 = 0;
        pp1 = 0;
    }



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

    int getPriorProbability(int val){
        int count = 0;
        for (ArrayList<Integer> outer : trainData){
            if(outer.get(0).equals(val)){count++;}
        }
        return count;
    }

    int getTotalDataRows(ArrayList<ArrayList<Integer>> list){
        return list.size();
    }

    //Columns
    int getTotalDataFeatures(ArrayList<ArrayList<Integer>> list){
        return list.get(0).size()-1;
    }


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

    public ArrayList<Integer> calculateLikelihood(int outcome, int given){
        ArrayList<Integer> arrayList = new ArrayList<Integer>();
        //System.out.println("Numfeatures: "+numfeatures);
        int count;
        for(int i = 1; i< numfeatures+1; i++){
            //System.out.print("i: " + i +" ||" );
            count =0;
            for(int j = 0; j< sample;j++){
                if(trainData.get(j).get(0).equals(outcome) && trainData.get(j).get(i).equals(given)){
                    count++;
                }
            }
            //System.out.print("count: " + count);
            //System.out.println();
            arrayList.add(count);

        }
        return arrayList;
    }


    public void train(){
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

    public boolean calculateEntry(ArrayList<Integer> entry){
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

        if (result == 0){
            return abnormalProb > normalProb;
        }else{
            return normalProb > abnormalProb;
        }
    }

    public void detect(){
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

    public void displayTrainData(){
        displayList(trainData);
    }
    public void displayTestData(){
        displayList(testData);
    }

    public ArrayList<ArrayList<Integer>> getTrainData(){
        return trainData;
    }

    public ArrayList<ArrayList<Integer>> getTestData(){
        return testData;
    }


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
        //detective.displayTrainData();

        detective.train();
        detective.detect();



        /*
        ArrayList<Integer> aListt = detective.calculateLikelihood(0,1);


        //int [] array = detective.getNormalizingConstant(0);


        System.out.println("--------------------------------------------");

        for(Integer val :aListt){
            System.out.print(val + ",");
        }
        */

        /*
        for(int i = 0; i<array.length; i++){
            System.out.print(array[i]+ ",");
        }
        */

        System.out.println();
    }

}
