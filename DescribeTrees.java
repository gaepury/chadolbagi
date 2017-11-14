package com.example.g.cardet;


import android.util.Log;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;

public class DescribeTrees {
	// method to take the txt fle as input and pass those values to random
	// forests
	BufferedReader BR = null;
	String path;

	public DescribeTrees(String path) {
		this.path = path;
	}

	public ArrayList<int[]> CreateInput(String path) {

		ArrayList<int[]> DataInput = new ArrayList<int[]>();

		try {

			String sCurrentLine;
			BR = new BufferedReader(new FileReader(path));

			while ((sCurrentLine = BR.readLine()) != null) {
				if (sCurrentLine != null) {
					String[] split = sCurrentLine.split(",");
					int[] Data =new int[split.length];
					for (int i = 0; i < split.length; i++) {
						Data[i]=Integer.parseInt(split[i]);
					}
					DataInput.add(Data);
												// t=0;t<DataInput.get(0).length;t++){System.out.print(DataInput.get(0)[t]+",");}System.out.println("");
				}
			}

		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {
				if (BR != null)
					BR.close();
			} catch (IOException ex) {
				ex.printStackTrace();
			}
		}
		return DataInput;
	}
	public ArrayList<int[]> CreateInput(File saveFile) {

		ArrayList<int[]> DataInput = new ArrayList<int[]>();

		try {

			String sCurrentLine;
			BR = new BufferedReader(new FileReader(saveFile));

			while ((sCurrentLine = BR.readLine()) != null) {
				if (sCurrentLine != null) {
					String[] split = sCurrentLine.split(",");
					int[] Data =new int[split.length]; //split.length = 8;
					for (int i = 0; i < split.length; i++) {
						Data[i]=Integer.parseInt(split[i]);
					}
					DataInput.add(Data);
					// t=0;t<DataInput.get(0).length;t++){System.out.print(DataInput.get(0)[t]+",");}System.out.println("");
				}
			}

		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {
				if (BR != null)
					BR.close();
			} catch (IOException ex) {
				ex.printStackTrace();
			}
		}
		return DataInput;
	}
	public ArrayList<int[]> CreateInput(InputStream path) {

		ArrayList<int[]> DataInput = new ArrayList<int[]>();

		try {

			String sCurrentLine;
			BR = new BufferedReader(new InputStreamReader(path));

			while ((sCurrentLine = BR.readLine()) != null) {
				if (sCurrentLine != null) {
					String[] split = sCurrentLine.split(",");
					int[] Data =new int[split.length];
					for (int i = 0; i < split.length; i++) {
						Data[i]=Integer.parseInt(split[i]);
					}

					DataInput.add(Data);
					// for(int
					// t=0;t<DataInput.get(0).length;t++){System.out.print(DataInput.get(0)[t]+",");}System.out.println("");
				}
			}

		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {
				if (BR != null)
					BR.close();
			} catch (IOException ex) {
				ex.printStackTrace();
			}
		}
		return DataInput;
	}
}
