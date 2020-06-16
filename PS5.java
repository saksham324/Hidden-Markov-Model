/*
* CS 10, Dartmouth Winter 2020
* Problem Set 5 Submission
* @author Saksham Arora
* @author Egemen Sahin
* Single Shared Code
*
 */



import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.lang.Math;

public class PS5 {
    public static Map<String, Map<String, Double>> transitionMap, observationMap;
    String line;
    private static String start = "#";
    //value of probability for an unseen observation
    double U = -100;


    // method to get the data from a text file and return a list containing every sentence
    public List<String[]> inputHolder(String pathname) throws IOException {
        BufferedReader input = new BufferedReader(new FileReader(pathname));

        List<String[]> inputs = new ArrayList<String[]>();

        while ((line = input.readLine()) != null) {
            // convert it to lower case
            line.toLowerCase();
            String[] s = line.split(" ");
            inputs.add(s);
        }

        input.close();

        return inputs;
    }

    // helper method to reduce repetition of code for creating transitionMap and observationMap
    public void helper (Map<String, Map<String, Double>> map, String key, String value) {
        // if map doesnt contain key
        if (!map.containsKey(key)) {

            // add the key and initialize a new, empty HashMap as the value
            map.put(key, new HashMap<String, Double>());

            // put the value in the correspoding map with a frequency of 1.0
            map.get(key).put(value, 1.0);

            // else if the map has the key, but the map doesn't have the value, add the value in the map with a freq of 1.0
        } else if (!map.get(key).containsKey(value)) {
            map.get(key).put(value, 1.0);

            // or else just increment the frequency of the value seen
        } else {
            map.get(key).put(value, map.get(key).get(value) + 1.0);
        }
    }


    // method to normalize the probability values using Natural Log
    public void normalizer (Map<String, Map<String, Double>> map) {

        // calculate the size of the map, or the total number of observations
        for (String key : map.keySet()) {
            double size = 0;
            for (String key2 : map.get(key).keySet())
            {
                size += map.get(key).get(key2);
            }

            // divide the frequency by the size and take thr natural log and add that as the value
            for (String key2 : map.get(key).keySet())
            {
                double prob = Math.log((map.get(key).get(key2)) / size);
                map.get(key).put(key2, prob);
            }
        }
    }


    // method to train model on a given set of training data. Takes in sentenceList and tagsList as parameters

    public void trainingModel(List<String[]> sentencesList, List<String[]> tagsList) {

        // initialize transitionMap and observationMap as new, empty HashMaps
        transitionMap = new HashMap<String, Map<String, Double>>();
        observationMap = new HashMap<String, Map<String, Double>>();


        // Setting transitionMap and observationMap
        for (int sentenceI = 0; sentenceI < sentencesList.size(); sentenceI++) {
            // for every sentence of words and every sentence of tags, take out the individual words and tags.
            String[] words = sentencesList.get(sentenceI);
            String[] tags = tagsList.get(sentenceI);

            // starting with hash
            String currState = start;

            // for every word in words and every corresponding tag in tags
            for(int i = 0; i < words.length; i++) {
                String word = words[i];
                String nextState = tags[i];

                // create the transitionMap and observationMap
                helper(transitionMap, currState, nextState);
                helper(observationMap, nextState, word);


                // set currState to nextState
                currState = nextState;
            }
        }

        // normalize the values of the observations and the transitions
        normalizer(transitionMap);
        normalizer(observationMap);
    }


    // method to perform viterbi decoding on a given sentence passed as parameter
    public String[] viterbi(String[] sentence) {

        // initialize currStates as a set
        Set<String> currStates = new HashSet<String>();

        // initialize currscores as a Map of currstates and the subsequent scores
        Map<String, Double> currScores = new HashMap<>();

        // initialize backTracker as a list which we will use to backTrack to the tags for each word
        List<Map<String, String>> backTracker = new ArrayList<Map<String, String>>();

        // add start in the currStates with a corresponding value of 0.0
        currStates.add(start);
        currScores.put(start, 0.0);

        // now for every word or observation in the sentence
        for (int i = 0; i < sentence.length; i++){

            // create a new set of nextStates and a map for storing nextScores
            Set<String> nextStates = new HashSet<String>();
            Map<String, Double> nextScores = new HashMap<>();

            // the current obsersation is the ith word in the sentence
            String currWord = sentence[i];

            // add this to the backTracker list and initialize a new Map as the value in the List
            backTracker.add(new HashMap<String, String>());	// add a back tracing map for the corresponding observation

            // for every currstate in currstates
            for (String currState: currStates)
            {
                // if the transitionMap contains current state
                if (transitionMap.containsKey(currState)) // if doesn't have it then nothing to add.
                {
                    // loop over all possible transitions of the currState, known as nextState
                    for (String nextState : transitionMap.get(currState).keySet())
                    {
                        // compute nextScore
                        double nextScore = currScores.get(currState) + transitionMap.get(currState).get(nextState);

                        // if the observation has never been seen before than add U to the score
                        if (!observationMap.get(nextState).containsKey(currWord)) {
                            nextScore += U;
                        }

                        // else add the observation score of the observation
                        else {
                            nextScore += observationMap.get(nextState).get(currWord);
                        }

                        // now if nextState doesn't have a nextScore yet, or if the nextScore is greater than the current
                        // nextScore then add/change the value of nextScore
                        if (!nextScores.containsKey(nextState) || nextScore > nextScores.get(nextState)) {
                            // add nextState and change nextScore value
                            nextStates.add(nextState);
                            nextScores.put(nextState, nextScore);

                            // keep track of previous state that produces the best score upon transition and observation
                            backTracker.get(i).put(nextState, currState);

                        }
                    }
                }
            }

            // set currStates and currScores to be nextStates and nextScores
            currStates = nextStates;
            currScores = nextScores;
        }



        // code for computing bestPath as per the highest probability of event occuring

        // start by setting the highest prob ending to be any one of the currStates
        String maxProbEnding = currStates.iterator().next();

        // loop over all current states
        for(String currEnding : currStates)
        {
            // if the new probability is higher than the current, then change the max prob ending
            if(currScores.get(currEnding) > currScores.get(maxProbEnding)) { maxProbEnding = currEnding; }
        }

        // compute bestPath by going backward through backTracker
        String curr = maxProbEnding;
        String[] bestPath = new String[sentence.length];

        // start from the end go back to the start.
        for(int i = sentence.length - 1; i >= 0; i--)
        {
            bestPath[i] = curr;
            curr = backTracker.get(i).get(curr);
        }

        return bestPath;

    }

    // method for consoleTesting
    public void consoleTest() throws IOException {
        Scanner in = new Scanner(System.in);
        System.out.println("Enter a sentence to be decoded : ");
        String input = in.nextLine();
        String[] words = input.split(" ");

        String[] path = viterbi(words);
        for (int i=0; i<path.length; i++) {
            System.out.println(words[i] + " -------- " + path[i]);
        }

    }


    // method for performance Testing which return the percentage accuracy of the model
    public void performanceTesting(List<String[]> input, List<String[]> target){
        double perf = 0;
        double numTags = 0;
        for (int sentenceI = 0; sentenceI < input.size(); sentenceI++) {
            String[] words = input.get(sentenceI);
            String[] tags = target.get(sentenceI);

            String [] path = viterbi(words);
            for(int i = 0; i < path.length; i++){
                if(path[i].equals(tags[i])) { perf++; }
                numTags ++;
            }
        }

        System.out.println("Model accuracy is " + perf/numTags * 100 + " % with " + (int)perf + " correct and " + (int) (numTags - perf) + " wrong. ");


    }



    public static void main(String[] args) throws IOException {

        PS5 simpleTestMap = new PS5();

        // train model on simple train sentences and tags
        List<String[]> sentences =  simpleTestMap.inputHolder("inputs/simple-train-sentences.txt");
        List<String[]> tags = simpleTestMap.inputHolder("inputs/simple-train-tags.txt");
        simpleTestMap.trainingModel(sentences, tags);

        // test model on simple test and compare it with testTags
        List<String[]> testSentences =  simpleTestMap.inputHolder("inputs/simple-test-sentences.txt");
        List<String[]> testTags =  simpleTestMap.inputHolder("inputs/simple-test-tags.txt");

        simpleTestMap.performanceTesting(testSentences, testTags);

//        // testing model from programming drill sentences and tags
//        List<String[]> sentencesTest =  simpleTestMap.inputHolder("inputs/testwords.txt");
//        List<String[]> tagsTest = simpleTestMap.inputHolder("inputs/testtags.txt");
//        simpleTestMap.trainingModel(sentencesTest, tagsTest);


        // testing model on brownTest
        PS5 brownTestMap = new PS5();
        List<String[]> brownSentences =  brownTestMap.inputHolder("inputs/brown-train-sentences.txt");
        List<String[]> brownTags = brownTestMap.inputHolder("inputs/brown-train-tags.txt");
        brownTestMap.trainingModel(brownSentences, brownTags);
//
//        List<String[]> brownTestSentences =  brownTestMap.inputHolder("inputs/brown-test-sentences.txt");
//        List<String[]> brownTestTags =  brownTestMap.inputHolder("inputs/brown-test-tags.txt");
//
//
//        brownTestMap.performanceTesting(brownTestSentences, brownTestTags);

//        brownTestMap.U = -8;
//        brownTestMap.performanceTesting(brownTestSentences, brownTestTags);

        //console Testing
        brownTestMap.consoleTest();





    }
}
