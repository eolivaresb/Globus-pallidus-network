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
//////////////////////////////////////////////////////////
// Integration Parameters
const double tstop = 1800.0, tadapt = 3.0, StepSize = 0.0001;
// Network parameters
const int N = 1000;
const int n = 10; // numer of synapses going out and in of a neuron = 10 out of 1000
// Files for neurons and network configuration
char path_to_files[] = "./../../simulation_files/";
ifstream neurons_file(std::string(path_to_files) + "neurons.dat");
ifstream Connectivity_file(std::string(path_to_files) + "network_random.dat");
ifstream prc_file(std::string(path_to_files) + "diff_prc.dat");
ifstream volt_file(std::string(path_to_files) + "volt.dat");

//////////////////////////////////////////////////////////
int main (int argc, const char* argv[]){
    RandomState rs;
    long randomseed = static_cast<long>(time(NULL));
    rs.SetRandomSeed(randomseed);
    cout.precision(10);

//////////////////////////////////////////////////////////
///////////////// Construct network 
//////////////////////////////////////////////////////////
    NervousSystem gp;
    gp.SetCircuitSize(N, 3*n); // Accept until 3n = 30 incomming synapses.
    // Load mean Voltage trace
    gp.LoadVoltage(volt_file);
    gp.IpscStd = 1750;
    // load neurons propierties
    TVector<double> NeuronsProp(1, 2*N);
    neurons_file >> NeuronsProp;
    for (int i=1; i<=N; i++){
        gp.SetNeuronW(i,NeuronsProp[2*i-1]);
        gp.SetNeuronCV(i, NeuronsProp[2*i]);
    }
    // load network connectivity 
    TVector<double> Connectivity(1, 2*n*N);
    Connectivity_file >> Connectivity;
    if (Connectivity[1] != 0){
        for (int i=1; i<=n*N; i++){
            gp.SetChemicalSynapseWeight(Connectivity[2*i-1], Connectivity[2*i], 1);
        }
    }
    // load neurons PRCs
    TVector<double> prc_params(1, 7*N);
    prc_file >> prc_params;
    for (int i=1; i<=N; i++){
        for (int p=1; p<=7; p++){
            gp.PrcNeuron[i].PrcParam[p] = prc_params[7*(i-1)+p];
        }
    }
    prc_file.close();
////////////////////////////////////////////////////////// 
///////////////   Simulation  
//////////////////////////////////////////////////////////
    // Inicialization
    gp.RandomizeCircuitPhase(0.0, 1.0, rs);
    int phase_bins = 1000;
    int phase_indx;
    TMatrix<int> phase_dist(1,N, 1,phase_bins);
    
    ofstream conductances("cond.dat");
//////////////////////////////////////////////
    // Run Simulation
//////////////////////////////////////////////
    //Network adaptation
    gp.SetSaveSpikes(false);
    for (double t = -tadapt; t < 0; t += StepSize){
        gp.EulerStep(StepSize, t, rs);
    }
    
    gp.SetSaveSpikes(true);
    for (double t = 0.0; t <tstop; t += StepSize){
        gp.EulerStep(StepSize, t, rs);
        for (int i=1; i<=N; i++){
            phase_indx = (int(phase_bins*gp.NeuronPhase(i))<0)?0:int(phase_bins*gp.NeuronPhase(i));
            phase_dist(i, 1+phase_indx)+=1;
        }
        if (t>10 and t< 14){
            for (int i=1; i<=N; i++){conductances << gp.Conductance[i] << " ";}
            conductances << endl;
        }
    }
    ofstream phase_densities("phase_densities.dat");
    
    phase_densities << phase_dist;
    phase_densities.close();
    conductances.close();
//////////////////////////////////////////////
}
