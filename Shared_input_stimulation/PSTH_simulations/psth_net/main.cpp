// =============================================================
// =============================================================
#include <iostream>
#include <iomanip>  // cout precision
#include <math.h>
#include "VectorMatrix.h"
#include "NervousSystem.h"
using namespace std;

//////////////////////////////////////////////////////////
///////////////// Simulation and network configuration
// Integration Parameters
const double ttot = 6000, StepSize = 0.0001;
// Network parameters
const int N = 1000;
const int n = 10; // numer of synapses going out and in of a neuron = 10 out of 1000
// Files for neurons and network configuration
char path_to_files[] = "../../simulation_files/";
ifstream neurons_file(std::string(path_to_files) + "neurons.dat");
ifstream Connected_file(std::string(path_to_files) + "network_small.dat");
ifstream prc_file(std::string(path_to_files) + "diff_prc.dat");
ifstream volt_file(std::string(path_to_files) + "volt.dat");
// Stimuli parameters
const double td = 3.5/1000;
double Erev = -0.073;
double gmax = 250;
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
int main (int argc, const char* argv[]){
/////////////////      read gmax and Erev from terminal  ////////////////////////////
    gmax = atof(argv[1]);
//////////////////////////////////////////////////////////
    RandomState rs;
    long randomseed = static_cast<long>(time(NULL));
    cout << "randomseed  " << randomseed << endl;
    rs.SetRandomSeed(randomseed);
    cout.precision(10);
//////////////////////////////////////////////////////////
    TVector<double> NeuronsProp(1, 2*N);
    neurons_file >> NeuronsProp;
    neurons_file.close();

    TVector<double> Connected(1, 2*n*N);
    Connected_file >> Connected;
    Connected_file.close();

    TVector<double> prc_params(1, 7*N);
    prc_file >> prc_params;
    prc_file.close();

///////////////// Construct connected network
    NervousSystem gp;           // Connected GPe
    gp.SetCircuitSize(N, 3*n);

    // Load mean Voltage trace
    gp.LoadVoltage(volt_file);
    // Set conductance amplitud variability (Std normal distribution pS)
    gp.IpscStd = 1750;
    // load neurons propierties
    for (int i=1; i<=N; i++){
        gp.SetNeuronW(i,NeuronsProp[2*i-1]);
        gp.SetNeuronCV(i, 0.05);
    }
    // load network connectivity
    for (int i=1; i<=n*N; i++){
        gp.SetChemicalSynapseWeight(Connected[2*i-1], Connected[2*i], 1);
    }
    // load neurons PRCs
    for (int i=1; i<=N; i++){
        for (int p=1; p<=7; p++){
            gp.PrcNeuron[i].PrcParam[p] = prc_params[7*(i-1)+p];
        }
    }
    //////////////////////////////////////////////
    //////////////////////////////////////////////
        // Inicialization
        gp.RandomizeCircuitPhase(0.0, 1.0, rs);
        //////////////////////////////////////////////
        ofstream First_spk_file("First_spk.dat", ios::out|ios::binary);
        //Record and generate stimuli time
        double gsyn = 0;
        double tmin = 0.6, tvar = 0.6;
        int nrep = 10000;
        double tinit = -0.025, tend = 0.175;
        double tadapt = tmin + rs.UniformRandom(0, tvar);
        //////////////////////////////////////////////
        int pdens_bins = 250;
        int time_bins = 1+(tend-tinit)*1000;//200
        TMatrix<int> pdens(1, pdens_bins, 1, N*time_bins);
        pdens.FillContents(0);
        TMatrix<int> conductances(1, N, 1,10*time_bins);
        conductances.FillContents(0);
        int tindx, tindx_cond;
        //////////////////////////////////////////////
        for (int rep = 1; rep<=nrep; rep++){
            //////////////////////////////////////////////
            gp.SetSaveSpikes(true);
            for (double t = tinit; t <0-StepSize/10; t += StepSize){
                tindx = int(1000*(t-tinit+StepSize/10)+1);
                tindx_cond = int(10000*(t-tinit+1.1*StepSize)+1);
                for (int i=1; i<=N; i++)  {
                    conductances[i][tindx_cond] += gp.Conductance[i];
                }
                gp.EulerStep(StepSize, t, rs);
                for (int i=1; i<=N; i++)  {
                    pdens[int(1+gp.NeuronPhase(i)*pdens_bins)][(i-1)*time_bins+tindx] +=1;
                }
            }
            //////////////////////////////////////////////
            gp.TimeToFirstSpike.FillContents(-1); // rest vector with time of the first spike after stim
            for (double t = 0; t <tend-StepSize/10; t += StepSize){
                tindx = int(1000*(t-tinit+StepSize/10)+1);
                tindx_cond = int(10000*(t-tinit+1.1*StepSize)+1);
                gsyn = gmax*(exp(-t/td));
                for (int i=1; i<=N; i++)  {
                    gp.externalinputs[i] = gsyn * (Erev - gp.VoltEval(gp.NeuronPhase(i)));
                    conductances[i][tindx_cond] += gp.Conductance[i];
                }
                gp.EulerStep(StepSize, t, rs);
                for (int i=1; i<=N; i++)  {
                    pdens[int(1+gp.NeuronPhase(i)*pdens_bins)][(i-1)*time_bins+tindx] +=1;
                }
            }
            //////////////////////////////////////////////
            tadapt = tmin + rs.UniformRandom(0, tvar);
            gp.SetSaveSpikes(false); // do not save adaptation time spikes (way too big files otherwise)
            for (double t = tend; t <(tend+tadapt); t += StepSize){
                gp.EulerStep(StepSize, t, rs);
            }
            /// Save time for the first spike
            for (int i=1; i<=N; i++)  {
                First_spk_file.write((char*)&(gp.TimeToFirstSpike(i)), sizeof(float));
            }
        }
        //////////////////////////////////////////////
        ofstream pdens_file("pdens.dat", ios::out|ios::binary);
        for (int i=1; i<=pdens_bins; i++)  {
            for (int j=1; j<=N*time_bins; j++)  {
                pdens_file.write((char*)&pdens[i][j], sizeof(int));
            }
            pdens_file << endl;
        }
        ofstream cond_file("cond.dat");
        for (int i=1; i<=N; i++)  {
            for (int j=1; j<=10*time_bins; j++)  {
                cond_file << conductances[i][j] << " ";
            }
            cond_file << endl;
        }
    }
