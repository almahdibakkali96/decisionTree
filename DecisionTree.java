
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import java.util.Random;

import java.lang.*;

import java.io.IOException;
import java.util.Scanner;
import java.io.File;
import java.io.BufferedReader;
import java.io.FileReader;

import com.toshiba.mwcloud.gs.Collection;
import com.toshiba.mwcloud.gs.ColumnInfo;
import com.toshiba.mwcloud.gs.Container;
import com.toshiba.mwcloud.gs.ContainerInfo;
import com.toshiba.mwcloud.gs.GSType;
import com.toshiba.mwcloud.gs.GridStore;
import com.toshiba.mwcloud.gs.GridStoreFactory;
import com.toshiba.mwcloud.gs.Query;
import com.toshiba.mwcloud.gs.Row;
import com.toshiba.mwcloud.gs.RowSet;



import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;




public class Select {

    public static void main(String[] args){
        try {

// Manage connection to GridDB
            Properties prop = new Properties();
            prop.setProperty("notificationAddress", "239.0.0.1");
            prop.setProperty("notificationPort", "31999");
            prop.setProperty("clusterName", "cluster");
            prop.setProperty("database", "public");
            prop.setProperty("user", "admin");
            prop.setProperty("password", "admin");
//Get Store and Container
            GridStore store = GridStoreFactory.getInstance().getGridStore(prop);
            
            store.getContainer("newContainer");

            String containerName = "mContainer";
        
// Define ontainer schema and columns
        ContainerInfo containerInfo = new ContainerInfo();
        List<ColumnInfo> columnList = new ArrayList<ColumnInfo>();
        columnList.add(new ColumnInfo("key", GSType.INTEGER));
        columnList.add(new ColumnInfo("slenght", GSType.FLOAT));
        columnList.add(new ColumnInfo("swidth", GSType.FLOAT));
        columnList.add(new ColumnInfo("plenght", GSType.FLOAT));
        columnList.add(new ColumnInfo("pwidth", GSType.FLOAT));
        columnList.add(new ColumnInfo("irisclass", GSType.STRING));

        containerInfo.setColumnInfoList(columnList);
        containerInfo.setRowKeyAssigned(true);
        Collection<Void, Row> collection = store.putCollection(containerName, containerInfo, false);
        List<Row> rowList = new ArrayList<Row>();

// Handlig Dataset and storage to GridDB
            File data = new File("/home/ubuntu/griddb/gsSample/iris.csv");
            Scanner sc = new Scanner(data);  
            sc.useDelimiter("\n");
            while (sc.hasNext())  //returns a boolean value  
            {  
                int i = 0;
            Row row = collection.createRow();

            String line = sc.next();
            String columns[] = line.split(",");
            float slenght = Float.parseFloat(columns[0]);
            float swidth = Float.parseFloat(columns[1]);
            float plenght = Float.parseFloat(columns[2]);
            float pwidth = Float.parseFloat(columns[3]);
            String irisclass = columns[4];
                
            row.setInteger(0,i);
            row.setFloat(1,slenght );
            row.setFloat(2, swidth);
            row.setFloat(3, plenght);
            row.setFloat(4, pwidth);
            row.setString(5, irisclass);

            rowList.add(row);
    
            i++;
        }   
        
// Retrieving data from GridDB
        
        Container<?, Row> container = store.getContainer(containerName);

        if ( container == null ){
            throw new Exception("Container not found.");
        }
        Query<Row> query = container.query("SELECT * ");
        RowSet<Row> rs = query.fetch();


    // BufferedReader bufferedReader= new BufferedReader(new FileReader(rs));
    // Instances datasetInstances= new Instances(bufferedReader);
  
  
    DataSource source = new DataSource("/home/ubuntu/griddb/gsSample/iris.csv");
    Instances datasetInstances = source.getDataSet();
    
        datasetInstances.setClassIndex(0);

        String[] options = new String[4];
        options[0] = "-C";
        options[1] = "0.25";
        options[2] = "-M";
        options[3] = "30";
         
        J48 mytree = new J48();
        mytree.setOptions(options);
        datasetInstances.setClassIndex(datasetInstances.numAttributes()-1);

        mytree.buildClassifier(datasetInstances);
    
        //Perform Evaluation 

    Evaluation eval = new Evaluation(datasetInstances);
    eval.crossValidateModel(mytree, datasetInstances, 10, new Random(1));

    System.out.println(eval.toSummaryString("\n ****** J48 *****\n", true));

     // Print GridDB data
        while ( rs.hasNext() ) {
            Row row = rs.next();
            float slenght = row.getFloat(0);
            float swidth = row.getFloat(1);
            float plenght = row.getFloat(2);
            float pwidth = row.getFloat(3);
            String irisclass = row.getString(4);
            System.out.println(" slenght=" + slenght + ", swidth=" + swidth + ", plenght=" + plenght +", pwidth=" + pwidth+", irisclass=" + irisclass);
        }

    // Terminating process
  
        collection.put(rowList);
        sc.close();  //closes the scanner 
        rs.close();
        query.close();
        container.close();
        store.close();
        System.out.println("success!");          
            

        } catch ( Exception e ){
            e.printStackTrace();
        }
    }


}