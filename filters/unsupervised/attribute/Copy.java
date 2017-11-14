/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    Copy.java
 *    Copyright (C) 1999 University of Waikato, Hamilton, New Zealand
 *
 */


package com.example.g.cardet.filters.unsupervised.attribute;

import com.example.g.cardet.core.Attribute;
import com.example.g.cardet.core.Capabilities;
import com.example.g.cardet.core.Instance;
import com.example.g.cardet.core.DenseInstance;
import com.example.g.cardet.core.Instances;
import com.example.g.cardet.core.Option;
import com.example.g.cardet.core.OptionHandler;
import com.example.g.cardet.core.Range;
import com.example.g.cardet.core.RevisionUtils;
import com.example.g.cardet.core.SparseInstance;
import com.example.g.cardet.core.Utils;
import com.example.g.cardet.core.Capabilities.Capability;
import com.example.g.cardet.filters.Filter;
import com.example.g.cardet.filters.StreamableFilter;
import com.example.g.cardet.filters.UnsupervisedFilter;

import java.util.Enumeration;
import java.util.Vector;

/** 
 <!-- globalinfo-start -->
 * An instance filter that copies a range of attributes in the dataset. This is used in conjunction with other filters that overwrite attribute values during the course of their operation -- this filter allows the original attributes to be kept as well as the new attributes.
 * <p/>
 <!-- globalinfo-end -->
 * 
 <!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -R &lt;index1,index2-index4,...&gt;
 *  Specify list of columns to copy. First and last are valid
 *  indexes. (default none)</pre>
 * 
 * <pre> -V
 *  Invert matching sense (i.e. copy all non-specified columns)</pre>
 * 
 <!-- options-end -->
 *
 * @author Len Trigg (trigg@cs.waikato.ac.nz)
 * @version $Revision: 6997 $
 */
public class Copy 
  extends Filter 
  implements UnsupervisedFilter, StreamableFilter, OptionHandler {
  
  /** for serialization */
  static final long serialVersionUID = -8543707493627441566L;

  /** Stores which columns to copy */
  protected Range m_CopyCols = new Range();

  /**
   * Stores the indexes of the selected attributes in order, once the
   * dataset is seen
   */
  protected int [] m_SelectedAttributes;

  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  public Enumeration listOptions() {

    Vector newVector = new Vector(2);

    newVector.addElement(new Option(
              "\tSpecify list of columns to copy. First and last are valid\n"
	      +"\tindexes. (default none)",
              "R", 1, "-R <index1,index2-index4,...>"));
    newVector.addElement(new Option(
	      "\tInvert matching sense (i.e. copy all non-specified columns)",
              "V", 0, "-V"));

    return newVector.elements();
  }

  /**
   * Parses a given list of options. <p/>
   * 
   <!-- options-start -->
   * Valid options are: <p/>
   * 
   * <pre> -R &lt;index1,index2-index4,...&gt;
   *  Specify list of columns to copy. First and last are valid
   *  indexes. (default none)</pre>
   * 
   * <pre> -V
   *  Invert matching sense (i.e. copy all non-specified columns)</pre>
   * 
   <!-- options-end -->
   *
   * @param options the list of options as an array of strings
   * @throws Exception if an option is not supported
   */
  public void setOptions(String[] options) throws Exception {

    String copyList = Utils.getOption('R', options);
    if (copyList.length() != 0) {
      setAttributeIndices(copyList);
    }
    setInvertSelection(Utils.getFlag('V', options));
    
    if (getInputFormat() != null) {
      setInputFormat(getInputFormat());
    }
  }

  /**
   * Gets the current settings of the filter.
   *
   * @return an array of strings suitable for passing to setOptions
   */
  public String [] getOptions() {

    String [] options = new String [3];
    int current = 0;

    if (getInvertSelection()) {
      options[current++] = "-V";
    }
    if (!getAttributeIndices().equals("")) {
      options[current++] = "-R"; options[current++] = getAttributeIndices();
    }

    while (current < options.length) {
      options[current++] = "";
    }
    return options;
  }

  /** 
   * Returns the Capabilities of this filter.
   *
   * @return            the capabilities of this object
   * @see               Capabilities
   */
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();
    result.disableAll();

    // attributes
    result.enableAllAttributes();
    result.enable(Capability.MISSING_VALUES);
    
    // class
    result.enableAllClasses();
    result.enable(Capability.MISSING_CLASS_VALUES);
    result.enable(Capability.NO_CLASS);
    
    return result;
  }

  /**
   * Sets the format of the input instances.
   *
   * @param instanceInfo an Instances object containing the input instance
   * structure (any instances contained in the object are ignored - only the
   * structure is required).
   * @return true if the outputFormat may be collected immediately
   * @throws Exception if a problem occurs setting the input format
   */
  public boolean setInputFormat(Instances instanceInfo) throws Exception {

    super.setInputFormat(instanceInfo);
    
    m_CopyCols.setUpper(instanceInfo.numAttributes() - 1);

    // Create the output buffer
    Instances outputFormat = new Instances(instanceInfo, 0); 
    m_SelectedAttributes = m_CopyCols.getSelection();
    for (int i = 0; i < m_SelectedAttributes.length; i++) {
      int current = m_SelectedAttributes[i];
      // Create a copy of the attribute with a different name
      Attribute origAttribute = instanceInfo.attribute(current);
      outputFormat.insertAttributeAt((Attribute)origAttribute.copy("Copy of " + origAttribute.name()),
				     outputFormat.numAttributes());

    }

    // adapt locators
    int[] newIndices = new int[instanceInfo.numAttributes() + m_SelectedAttributes.length];
    for (int i = 0; i < instanceInfo.numAttributes(); i++)
      newIndices[i] = i;
    for (int i = 0; i < m_SelectedAttributes.length; i++)
      newIndices[instanceInfo.numAttributes() + i] = m_SelectedAttributes[i];
    initInputLocators(instanceInfo, newIndices);

    setOutputFormat(outputFormat);
    
    return true;
  }
  

  /**
   * Input an instance for filtering. Ordinarily the instance is processed
   * and made available for output immediately. Some filters require all
   * instances be read before producing output.
   *
   * @param instance the input instance
   * @return true if the filtered instance may now be
   * collected with output().
   * @throws IllegalStateException if no input format has been defined.
   */
  public boolean input(Instance instance) {

    if (getInputFormat() == null) {
      throw new IllegalStateException("No input instance format defined");
    }
    if (m_NewBatch) {
      resetQueue();
      m_NewBatch = false;
    }

    double[] vals = new double[outputFormatPeek().numAttributes()];
    for(int i = 0; i < getInputFormat().numAttributes(); i++) {
      vals[i] = instance.value(i);
    }
    int j = getInputFormat().numAttributes();
    for (int i = 0; i < m_SelectedAttributes.length; i++) {
      int current = m_SelectedAttributes[i];
      vals[i + j] = instance.value(current);
    }
    Instance inst = null;
    if (instance instanceof SparseInstance) {
      inst = new SparseInstance(instance.weight(), vals);
    } else {
      inst = new DenseInstance(instance.weight(), vals);
    }
    
    inst.setDataset(getOutputFormat());
    copyValues(inst, false, instance.dataset(), getOutputFormat());
    inst.setDataset(getOutputFormat());
    push(inst);
    return true;
  }

  /**
   * Returns a string describing this filter
   *
   * @return a description of the filter suitable for
   * displaying in the explorer/experimenter gui
   */
  public String globalInfo() {

    return "An instance filter that copies a range of attributes in the"
      + " dataset. This is used in conjunction with other filters that"
      + " overwrite attribute values during the course of their operation --"
      + " this filter allows the original attributes to be kept as well"
      + " as the new attributes.";
  }

  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String invertSelectionTipText() {
    return "Sets copy selected vs unselected action."
      + " If set to false, only the specified attributes will be copied;"
      + " If set to true, non-specified attributes will be copied.";
  }

  /**
   * Get whether the supplied columns are to be removed or kept
   *
   * @return true if the supplied columns will be kept
   */
  public boolean getInvertSelection() {

    return m_CopyCols.getInvert();
  }

  /**
   * Set whether selected columns should be removed or kept. If true the 
   * selected columns are kept and unselected columns are copied. If false
   * selected columns are copied and unselected columns are kept. <br>
   * Note: use this method before you call 
   * <code>setInputFormat(Instances)</code>, since the output format is
   * determined in that method.
   *
   * @param invert the new invert setting
   */
  public void setInvertSelection(boolean invert) {

    m_CopyCols.setInvert(invert);
  }

  /**
   * Get the current range selection
   *
   * @return a string containing a comma separated list of ranges
   */
  public String getAttributeIndices() {

    return m_CopyCols.getRanges();
  }

  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String attributeIndicesTipText() {
    return "Specify range of attributes to act on."
      + " This is a comma separated list of attribute indices, with"
      + " \"first\" and \"last\" valid values. Specify an inclusive"
      + " range with \"-\". E.g: \"first-3,5,6-10,last\".";
  }

  /**
   * Set which attributes are to be copied (or kept if invert is true)
   *
   * @param rangeList a string representing the list of attributes.  Since
   * the string will typically come from a user, attributes are indexed from
   * 1. <br>
   * eg: first-3,5,6-last<br>
   * Note: use this method before you call 
   * <code>setInputFormat(Instances)</code>, since the output format is
   * determined in that method.
   * @throws Exception if an invalid range list is supplied
   */
  public void setAttributeIndices(String rangeList) throws Exception {

    m_CopyCols.setRanges(rangeList);
  }

  /**
   * Set which attributes are to be copied (or kept if invert is true)
   *
   * @param attributes an array containing indexes of attributes to select.
   * Since the array will typically come from a program, attributes are indexed
   * from 0.<br>
   * Note: use this method before you call 
   * <code>setInputFormat(Instances)</code>, since the output format is
   * determined in that method.
   * @throws Exception if an invalid set of ranges is supplied
   */
  public void setAttributeIndicesArray(int [] attributes) throws Exception {

    setAttributeIndices(Range.indicesToRangeList(attributes));
  }
  
  /**
   * Returns the revision string.
   * 
   * @return		the revision
   */
  public String getRevision() {
    return RevisionUtils.extract("$Revision: 6997 $");
  }

  /**
   * Main method for testing this class.
   *
   * @param argv should contain arguments to the filter: use -h for help
   */
  public static void main(String [] argv) {
    runFilter(new Copy(), argv);
  }
}
