using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace WpfApp1
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        string tobiiStartTime;
        public string naosPathString;
        string tobiiFilePath;

        public string naosAdjustedOutputFilePath;
        public string tobbiTimeAdjustedOutputFilePath;
        public string syncronizedOutputFilePath;
        string tobiiTimestringConversion = "HH:mm:ss";
        string tobiiFileName = "";

        public class ListBoxViewModel : INotifyPropertyChanged
        {
            public event PropertyChangedEventHandler PropertyChanged;
            //List box content
            public ObservableCollection<string> _listBoxItems; // <-- Add this

            public ListBoxViewModel()
            {
                _listBoxItems = new ObservableCollection<string>();
            }
        }

        public ListBoxViewModel listboxViewModel;


        string connationPrev;
        string connationNext;
     
        public int offsetBarFileLength = 1;
        public MainWindow()
        {
            listboxViewModel = new ListBoxViewModel();
            InitializeComponent();
            trimmingPartsListBox.DataContext = listboxViewModel;
            trimmingPartsListBox.ItemsSource = listboxViewModel._listBoxItems;
        }

        private void tobeOffsetPath_TextChanged(object sender, TextChangedEventArgs e)
        {
            naosPathString = toBeOffsetPath.Text;
        }

        private void selectToBeOffset_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            if (openFileDialog.ShowDialog() == true)
            {
                toBeOffsetPath.Text = openFileDialog.FileName;
                naosPathString = toBeOffsetPath.Text;
                offsetBarFileLength = File.ReadLines(naosPathString).Count();
            }
        }
        
        void ReadAllData()
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            if (openFileDialog.ShowDialog() == true)
                toBeOffsetPath.Text = File.ReadAllText(openFileDialog.FileName);
        }
        System.Threading.Thread myThread;
        private void offsetTimeButton_Click(object sender, RoutedEventArgs e)
        {
            OffsetData();
            //myThread = new System.Threading.Thread(new
            // System.Threading.ThreadStart(OffsetData));
            //myThread.Start();
        }

        int counter = 0;

        void OffsetData()
        {
            CultureInfo provider = CultureInfo.InvariantCulture;

            string line;
            if (string.IsNullOrWhiteSpace(naosPathString))
                return;
            // Read the file and display it line by line.  
            System.IO.StreamReader file =
                new System.IO.StreamReader(naosPathString);
            string path = @"C:\Users\Nstov\Source\Repos\WpfApp1\WpfApp1\Output\naosTimeOffset_" + DateTime.Now.ToShortDateString() + " .txt";
            naosAdjustedOutputFilePath = path;
            // Create a file to write to.
            using (StreamWriter sw = File.CreateText(naosAdjustedOutputFilePath))
            {
                while ((line = file.ReadLine()) != null)
                {
                        if (!line.Contains("ID") && !string.IsNullOrWhiteSpace(line))
                        {
                            string[] sections = line.Split(',');
                        
                            DateTime readTime = DateTime.ParseExact(sections[1], "hh:mm:ss tt", provider);
                            readTime = readTime.Subtract(new TimeSpan(0,10,2));
                            sections[1] = readTime.ToLongTimeString();
                            string tempLine = null;
                            
                            for (int i = 0; i < sections.Length; i++)
                            {
                                tempLine += sections[i] + ",";
                            }
                            sw.WriteLine(tempLine);
                            System.Console.WriteLine(tempLine);
                        }

                    counter++;
                }
            }
            file.Close();
            newNaosFileTextBox.Text = naosAdjustedOutputFilePath;
            System.Console.WriteLine("There were {0} lines.", counter);
            // Suspend the screen.  
            System.Console.ReadLine();
            return;
        }

        private void selectTobbiFileButton_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            if (openFileDialog.ShowDialog() == true)
            {
                tobiiFilePathTextBox.Text = openFileDialog.FileName;
                tobiiFilePath = tobiiFilePathTextBox.Text;
                offsetBarFileLength = File.ReadLines(tobiiFilePath).Count();
            }
        }

        private void TextBox_TextChanged(object sender, TextChangedEventArgs e)
        {
            tobiiFilePath = tobiiFilePathTextBox.Text;
        }

        List<string> tempTobiiData;
        const int extraSampleTime = 1750 + 1; //Every 1750 sample we get one extra
                                              //Set milliseconds to proper times and remove events and fill gaps

        void ReadTobiiData()
        {
            CultureInfo provider = CultureInfo.InvariantCulture;

            DateTime result;
            bool parse = DateTime.TryParse(tobiiStartTime,out result);

            if (string.IsNullOrWhiteSpace(tobiiStartTime) || !parse)
                return;

            DateTime tobiiStartDateTime = DateTime.ParseExact(tobiiStartTime, "hh:mm:ss", provider);
            tempTobiiData = new List<string>();
            string line;
            if (string.IsNullOrWhiteSpace(tobiiFilePath))
                return;
            // Read the file and display it line by line.  
            System.IO.StreamReader file = new System.IO.StreamReader(tobiiFilePath);
            tobiiFileName = System.IO.Path.GetFileNameWithoutExtension(tobiiFilePath);

            string path = @"C:\Users\Nstov\Source\Repos\WpfApp1\WpfApp1\Output\tobiiTimeConverted_" + tobiiFileName + ".txt";

            tobbiTimeAdjustedOutputFilePath = path;
            double previousTime = 0;
            int duplicateCounter = 0;

            using (StreamWriter sw = File.CreateText(tobbiTimeAdjustedOutputFilePath))
            {
                while ((line = file.ReadLine()) != null)
                {
                    if (!line.Contains("Recording") && !string.IsNullOrWhiteSpace(line))
                    {
                        string[] sections = line.Split(tobiiSplitDelimiter.ToCharArray());
                        string tempLine = null;
                        
                        if (string.IsNullOrWhiteSpace(sections[9]))
                        {
                            //As the tobii eyetracker measures 50 samples per second, which is +20 milliseconds pr line in file
                            double milliseconds;
                            double.TryParse(sections[0], out milliseconds);

                            double offset = 20 - (milliseconds % 20);
                            if (offset == 20)
                                offset = 0;

                            milliseconds += offset;

                            //Removing duplicates
                            if (previousTime == milliseconds)
                            {
                                duplicateCounter++;
                            }

                            DateTime readTime;
                            string savedTime;
                            //Add missing Data if there is more than 20 milliseconds from last sample
                            offset = (milliseconds - previousTime) - 20;
                            if (offset > 20)
                            {
                                int missingLines = (int)offset / 20;
                                for (int i = 0; i < missingLines; i++)
                                {
                                    previousTime += 20;
                                    readTime = tobiiStartDateTime;
                                    readTime = readTime.AddMilliseconds(previousTime);
                                    //millisecondCount += 20;
                                    savedTime = readTime.ToString(tobiiTimestringConversion,
                                                        CultureInfo.InvariantCulture);
                                    string emptyLine = savedTime + ",0,0,0,0,0,0,0,0";

                                    tempTobiiData.Add(emptyLine);
                                    sw.WriteLine(emptyLine);
                                    Console.WriteLine(emptyLine);
                                    counter++;
                                }
                            }

                            previousTime = milliseconds;
                            readTime = tobiiStartDateTime;
                            readTime = readTime.AddMilliseconds(previousTime);
                            //millisecondCount += 20;
                            savedTime = readTime.ToString(tobiiTimestringConversion,
                                                CultureInfo.InvariantCulture);

                            sections[0] = savedTime;// previousTime.ToString();
                            for (int i = 0; i < sections.Length; i++)
                            {
                                //Replace commas with decimals
                                string item = sections[i];

                                if (string.IsNullOrWhiteSpace(item))
                                {
                                    item = 0.ToString();
                                }

                                double value;
                                bool parsed = double.TryParse(item, out value);

                                if (parsed)
                                {
                                    NumberFormatInfo nfi = new NumberFormatInfo();
                                    nfi.NumberDecimalSeparator = ".";
                                    sections[i] = value.ToString(nfi);
                                }
                                tempLine += sections[i] + ",";
                            }

                            tempTobiiData.Add(tempLine);
                            sw.WriteLine(tempLine);
                            Console.WriteLine(tempLine);
                            //}
                        }
                        else
                        {
                            //System.Console.WriteLine(line);
                        }
                    }
                    counter++;
                }
            }
            file.Close();
            newTobiiFileTextBox.Text = tobbiTimeAdjustedOutputFilePath;
            System.Console.WriteLine("There were {0} lines.", counter);
            System.Console.WriteLine("There were {0} duplicates.", duplicateCounter);

            // Suspend the screen.  
            System.Console.ReadLine();
        }

   
        //ConvertTimeValuesButton
        private void startSyncronizationButton_Click(object sender, RoutedEventArgs e)
        {
            ReadTobiiData();
            return;

            CultureInfo provider = CultureInfo.InvariantCulture;
            DateTime tobiiStartDateTime = DateTime.ParseExact(tobiiStartTime, "hh:mm:ss", provider);
         
            string line;
            if (string.IsNullOrWhiteSpace(tobiiFilePath))
                return;
            // Read the file and display it line by line.  
            System.IO.StreamReader file = new System.IO.StreamReader(tobiiFilePath);
            tobiiFileName = System.IO.Path.GetFileNameWithoutExtension(tobiiFilePath);

            string path = @"C:\Users\Nstov\Source\Repos\WpfApp1\WpfApp1\Output\tobiiTimeConverted_" + fileName + ".txt";

            tobbiTimeAdjustedOutputFilePath = path;

            using (StreamWriter sw = File.CreateText(tobbiTimeAdjustedOutputFilePath))
            {
                while ((line = file.ReadLine()) != null)
                {
                    if (!line.Contains("Recording") && !string.IsNullOrWhiteSpace(line) )
                    {
                        string[] sections = line.Split(tobiiSplitDelimiter.ToCharArray());
                        if (string.IsNullOrWhiteSpace(sections[9]))
                        {
                            //As the tobii eyetracker measures 50 samples per second, which is +20 milliseconds pr line in file
                            double milliseconds;
                            double.TryParse(sections[0], out milliseconds);
                            DateTime readTime = tobiiStartDateTime;
                            readTime = readTime.AddMilliseconds(milliseconds);
                            //millisecondCount += 20;
                            sections[0] = readTime.ToString(tobiiTimestringConversion,
                                                CultureInfo.InvariantCulture);
                            string tempLine = null;
                            //Concatenate array
                            for (int i = 0; i < sections.Length; i++)
                            {
                                //Replace commas with decimals
                                string item = sections[i];

                                if (string.IsNullOrWhiteSpace(item))
                                {
                                    item = 0.ToString();
                                }

                                double value;
                                bool parsed = double.TryParse(item, out value);
                                if (parsed)
                                {
                                    NumberFormatInfo nfi = new NumberFormatInfo();
                                    nfi.NumberDecimalSeparator = ".";
                                    sections[i] = value.ToString(nfi);
                                }
                                tempLine += sections[i] + ",";
                            }
                            sw.WriteLine(tempLine);
                            System.Console.WriteLine(tempLine);
                        }
                    }
                    counter++;
                }
            }
            file.Close();
            newTobiiFileTextBox.Text = tobbiTimeAdjustedOutputFilePath;
            System.Console.WriteLine("There were {0} lines.", counter);
            // Suspend the screen.  
            System.Console.ReadLine();
        }
        //Tobii start time texbox
        private void TextBox_TextChanged_1(object sender, TextChangedEventArgs e)
        {
            tobiiStartTime = ((TextBox)sender).Text;
        }
        string tobiiSplitDelimiter = "\t";

        private void alignDataButton_Click(object sender, RoutedEventArgs e)
        {
            CultureInfo provider = CultureInfo.InvariantCulture;

            //Find max time as that is where the syncrhonized file will begin
            System.IO.StreamReader tobiifile = new System.IO.StreamReader(tobbiTimeAdjustedOutputFilePath);
            System.IO.StreamReader naosfile = new System.IO.StreamReader(naosAdjustedOutputFilePath);

            string tobiiFirstLine;
            tobiiFirstLine = tobiifile.ReadLine();

            string naosFirstLine;
            naosFirstLine = naosfile.ReadLine();

            string[] tobiiSections = tobiiFirstLine.Split(',');
            string[] naosSections = naosFirstLine.Split(',');

            DateTime tobiiTime = DateTime.ParseExact(tobiiSections[0], tobiiTimestringConversion, provider);
            DateTime naosTime = DateTime.ParseExact(naosSections[1], "hh:mm:ss", provider);
            
            //Compare times
            bool naosFirst = tobiiTime > naosTime;
            System.Console.WriteLine(naosFirst);
            tobiiFileName = System.IO.Path.GetFileNameWithoutExtension(tobiiFilePath);
            //Search other file to find out where the first value at the min timestamp is
            //Then simply read concatenate and write values until either file runs out of values
            string alignedOutputPath = @"C:\Users\Nstov\Source\Repos\WpfApp1\WpfApp1\Output\SyncronizedOutput_" + fileName + " .txt";
            syncronizedOutputFilePath = alignedOutputPath;
            int x = 0;
            bool streaming = false;
            using (StreamWriter sw = File.CreateText(syncronizedOutputFilePath))
            {
                if (!naosFirst)
                {
                    while ((tobiiFirstLine = tobiifile.ReadLine()) != null && !streaming)
                    {
                        string[] tobiiSectionsTemp = tobiiFirstLine.Split(',');
                        DateTime tobiiTimeTemp = DateTime.ParseExact(tobiiSectionsTemp[0], tobiiTimestringConversion, provider);
                        if(tobiiTimeTemp >= naosTime)
                        {
                            streaming = true;
                        }
                    }
                    //Now i remove timestamps and concatenate
                    while (streaming && (tobiiFirstLine = tobiifile.ReadLine()) != null && (naosFirstLine = naosfile.ReadLine()) != null)
                    {
                        string newLine;
                        string[] tobiiSectionsTemp = tobiiFirstLine.Split(',');
                        string[] naosSectionsTemp = naosFirstLine.Split(',');
                        //Remove Naos TimeStamp and ID
                        string naosLine = null;
                        for (int i = 2; i < naosSectionsTemp.Length; i++)
                        {
                            naosLine += naosSectionsTemp[i] + ",";
                        }
                        naosLine = naosLine.TrimEnd(',');
                        //newLine = tobiiSectionsTemp[0] + "," + naosSectionsTemp[1];
                        newLine = tobiiFirstLine + naosLine;
                        sw.WriteLine(newLine);
                        System.Console.WriteLine(newLine);
                        counter++;
                    }
                }
                else
                {
                    while ((naosFirstLine = naosfile.ReadLine()) != null && !streaming)
                    {
                        string[] naosSectionsTemp = naosFirstLine.Split(',');
                        DateTime naosTimeTemp = DateTime.ParseExact(naosSectionsTemp[1], "hh:mm:ss", provider);
                        if (naosTimeTemp >= tobiiTime)
                        {
                            streaming = true;
                        }
                    }
                    //Now i remove timestamps and concatenate
                    while (streaming && (tobiiFirstLine = tobiifile.ReadLine()) != null && (naosFirstLine = naosfile.ReadLine()) != null)
                    {
                        string newLine;
                        string[] tobiiSectionsTemp = tobiiFirstLine.Split(',');
                        string[] naosSectionsTemp = naosFirstLine.Split(',');
                        string naosLine = null;
                        for (int i = 2; i < naosSectionsTemp.Length; i++)
                        {
                            naosLine += naosSectionsTemp[i] + ",";
                        }
                        naosLine = naosLine.TrimEnd(',');
                        newLine = tobiiFirstLine + naosLine;

                        sw.WriteLine(newLine);
                        System.Console.WriteLine(newLine);
                        counter++;
                    }
                }
            }
            mergedFilePathTextBox.Text = syncronizedOutputFilePath;
            naosfile.Close();
            tobiifile.Close();

            System.Console.WriteLine("There were {0} lines.", counter);
        }


        private void selectNewNaosFileButton_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            if (openFileDialog.ShowDialog() == true)
            {
                newNaosFileTextBox.Text = openFileDialog.FileName;
                naosAdjustedOutputFilePath = newNaosFileTextBox.Text;
                //offsetBarFileLength = File.ReadLines(naosAdjustedOutputFilePath).Count();
            }
        }

        private void selectNewTobiiFileButton_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            if (openFileDialog.ShowDialog() == true)
            {
                newTobiiFileTextBox.Text = openFileDialog.FileName;
                tobbiTimeAdjustedOutputFilePath = newTobiiFileTextBox.Text;
                //offsetBarFileLength = File.ReadLines(tobbiTimeAdjustedOutputFilePath).Count();
            }
        }

        string trimStartTime;
        string trimEndTime;
        //Syncronized filepath text box
        private void TextBox_TextChanged_2(object sender, TextChangedEventArgs e)
        {
            syncronizedOutputFilePath = ((TextBox)sender).Text;
        }

        private void selectMergedFileButton_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            if (openFileDialog.ShowDialog() == true)
            {
                mergedFilePathTextBox.Text = openFileDialog.FileName;
                syncronizedOutputFilePath = mergedFilePathTextBox.Text;
                //offsetBarFileLength = File.ReadLines(tobbiTimeAdjustedOutputFilePath).Count();
            }
        }

        private void trimStart_TextChanged(object sender, TextChangedEventArgs e)
        {
            trimStartTime = ((TextBox)sender).Text;
        }

        private void trimEnd_TextChanged(object sender, TextChangedEventArgs e)
        {
            trimEndTime = ((TextBox)sender).Text;
        }

        private void trimButton_Click(object sender, RoutedEventArgs e)
        {
            if(!string.IsNullOrWhiteSpace(trimStartTime) && !string.IsNullOrWhiteSpace(trimEndTime))
            TrimData();
        }
        int trimCount = 0;
        void TrimData()
        {
            counter = 0;

            if (string.IsNullOrWhiteSpace(syncronizedOutputFilePath))
                return;

            System.IO.StreamReader syncronizedFile = new System.IO.StreamReader(syncronizedOutputFilePath);
            CultureInfo provider = CultureInfo.InvariantCulture;

            if (string.IsNullOrWhiteSpace(tobiiStartTime))
                return;

            DateTime tobiiStartDateTime = DateTime.ParseExact(tobiiStartTime, "hh:mm:ss", provider);

            string[] startoffsetSection = trimStartTime.Split(':');
            string[] endoffsetSection = trimEndTime.Split(':');

            TimeSpan startOffset = new TimeSpan(0, int.Parse(startoffsetSection[0]),int.Parse(startoffsetSection[1]));
            TimeSpan endOffset = new TimeSpan(0, int.Parse(endoffsetSection[0]), int.Parse(endoffsetSection[1]));

            DateTime trimStart = tobiiStartDateTime.Add(startOffset);
            DateTime trimEnd = tobiiStartDateTime.Add(endOffset);

            tobiiFileName = System.IO.Path.GetFileNameWithoutExtension(tobiiFilePath);

            string trimmedOutputPath = @"C:\Users\Nstov\Source\Repos\WpfApp1\WpfApp1\Output\TrimmedOutput_"+ trimCount + "_" + fileName + " .txt";
            string line;
            List<string> trimData = listboxViewModel._listBoxItems.ToList();
            using (StreamWriter sw = new StreamWriter(trimmedOutputPath))
            {
                while (((line = syncronizedFile.ReadLine()) != null))
                {
                    string[] syncSection = line.Split(',');
                    DateTime curTime = DateTime.ParseExact(syncSection[0], "HH:mm:ss", provider);

                    List<DateTime> startTimes = new List<DateTime>();
                    List<DateTime> endTimes = new List<DateTime>();

                    bool skip = false;
                    string connationValue = ",-10";
                    int passedTrimmed = 0;
                    for (int i = 0; i < trimData.Count; i++)
                    {
                        string[] trimSections = trimData[i].Split(',');

                        startoffsetSection = trimSections[0].Split(':');
                        endoffsetSection = trimSections[1].Split(':');

                        startOffset = new TimeSpan(0, int.Parse(startoffsetSection[0]), int.Parse(startoffsetSection[1]));
                        endOffset = new TimeSpan(0, int.Parse(endoffsetSection[0]), int.Parse(endoffsetSection[1]));

                        trimStart = tobiiStartDateTime.Add(startOffset);
                        trimEnd = tobiiStartDateTime.Add(endOffset);

                        //trimStart = DateTime.ParseExact(trimSections[0], "HH:mm:ss", provider);
                        //trimEnd = DateTime.ParseExact(trimSections[1], "HH:mm:ss", provider);
                        //startTimes.Add(trimStart);
                        //endTimes.Add(trimEnd);

                        if((curTime > trimStart && curTime < trimEnd))
                        {
                            skip = true;
                            break;
                        }
                        else if (curTime < trimStart)
                        {
                            connationValue = "," + trimSections[2];
                            break;
                        }
                    }

                    if (!skip)
                    {
                        line += connationValue;
                        sw.WriteLine(line);
                        Console.WriteLine(line);
                        counter++;
                    }
                }
            }
            trimCount++;
            syncronizedFile.Close();
            trimmedFilePathTextBox.Text = trimmedOutputPath;
            syncronizedOutputFilePath = trimmedOutputPath;
            System.Console.WriteLine("There were {0} lines.", counter);
        }

        private void trimmedFilePath_TextChanged(object sender, TextChangedEventArgs e)
        {

        }

        private void addTrimSectionButton_Click(object sender, RoutedEventArgs e)
        {
            string trimSectionInfo = trimStartTime + "," + trimEndTime + "," + connationPrev + "," + connationNext;
            listboxViewModel._listBoxItems.Add(trimSectionInfo);
            //trimmingPartsListBox.DataSource = _listBoxItems;
        }

        private void connationPrevTextBox_TextChanged(object sender, TextChangedEventArgs e)
        {
            connationPrev = ((TextBox)sender).Text;
        }

        private void connationNextTextBox_TextChanged(object sender, TextChangedEventArgs e)
        {
            connationNext = ((TextBox)sender).Text;
        }

        private void trimmingPartsListBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            removalSlot = ((ListBox)sender).SelectedIndex;
        }
        string fileName;
        private void fileNameTextBox_TextChanged(object sender, TextChangedEventArgs e)
        {
            fileName = ((TextBox)sender).Text;
        }

        int removalSlot;
        private void removeContentButton_Click(object sender, RoutedEventArgs e)
        {
            if (removalSlot < listboxViewModel._listBoxItems.Count)
            {
                listboxViewModel._listBoxItems.RemoveAt(removalSlot);
            }
            //else
            //{
            //    listboxViewModel._listBoxItems.RemoveAt(listboxViewModel._listBoxItems.Count-1);
            //}
        }

        private void clearContentButton_Click(object sender, RoutedEventArgs e)
        {
            listboxViewModel._listBoxItems.Clear();
        }
    }
}
