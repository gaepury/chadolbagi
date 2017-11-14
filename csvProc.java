package com.example.g.cardet;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static java.util.Arrays.asList;

public class csvProc {
    public Double[][] proc(Double[] inList) {
        List<Double> input = Arrays.asList(inList);
        List<Double> record = new ArrayList<>();
        List<List<Double>> result = new ArrayList<>();
        Double[][] output;

        try {
            Double preSpd = 0.0;

            for(int z = 0; z < input.size(); z++){
                Double spd = input.get(z);

                if(spd < preSpd) {
                    record.add(spd);
                } else if(record.size() > 0) {
                    Double Sum = 0.0, SumDe = 0.0, MaxDe = 0.0;
                    for(int i = 0; i < record.size() - 1; i++) {
                        Sum += record.get(i);
                        SumDe += record.get(i + 1) - record.get(i);
                        if(MaxDe > record.get(i + 1) - record.get(i))
                            MaxDe = record.get(i + 1) - record.get(i);
                    }
                    Sum += record.get(record.size() - 1);

                    Double[] calc = new Double[6];
                    calc[0] = record.get(0);
                    calc[1] = record.get(record.size() - 1);
                    calc[2] = SumDe / record.size();
                    calc[3] = MaxDe;
                    calc[4] = Sum / 3.6;
                    calc[5] = (double)record.size();
                    result.add(asList(calc));
                    record.clear();
                }

                preSpd = spd;
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        output = new Double[result.size()][6];
        int i = 0;
        for (List<Double> nestedList : result)
            output[i++] = nestedList.toArray(new Double[nestedList.size()]);

        return output;
    }
}
