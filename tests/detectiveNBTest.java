import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class detectiveNBTest {

    @Test
    void testGetPriorProbability(){
        detectiveNB testcase = new detectiveNB("data/spect-orig.train.csv", "data/spect-orig.test.csv");
        assertEquals(40,testcase.getPriorProbability(1));
        assertEquals(40, testcase.getPriorProbability(0));
    }

    @Test
    void testGetTotalDataRows(){
        detectiveNB testcase = new detectiveNB("data/spect-orig.train.csv", "data/spect-orig.test.csv");
        assertEquals(80,testcase.getTotalDataRows(testcase.getTrainData()));
        assertEquals(187,testcase.getTotalDataRows(testcase.getTestData()));
    }

}