//
// Example Checkpoint - can be compiled anywhere
//
// Must have PIN_HOME environment variable set
// 

#include "pin.H"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <map>
#include <unistd.h> // for pid
#include <cstdlib>

/* ===================================================================== */
/* Print Help Message                                                    */
/* ===================================================================== */
static INT32 Usage()
{
    cerr << "This pin tool collects a profile of jump/branch/call instructions for an application\n";

    cerr << KNOB_BASE::StringKnobSummary();

    cerr << endl;
    return -1;
}
/* ===================================================================== */

/* ===================================================================== */
/* Commandline Switches */
/* ===================================================================== */

KNOB<string> KnobOutputFile(KNOB_MODE_WRITEONCE,         "pintool",
                            "o", "output.out", "specify profile file name");

KNOB<BOOL>   KnobPid(KNOB_MODE_WRITEONCE,                "pintool",
                            "i", "0", "append pid to output");

KNOB<UINT64> KnobBranchLimit(KNOB_MODE_WRITEONCE,        "pintool",
                            "l", "0", "set limit of dynamic branches simulated/analyzed before quit");

/* ===================================================================== */
/* Global Variables */
/* ===================================================================== */
UINT64 CountSeen = 0;
UINT64 CountTaken = 0;
UINT64 CountCorrect = 0;
UINT64 CountReplaced = 0;

/* ===================================================================== */
/* Branch predictor                                                      */
/* ===================================================================== */
UINT64 mask = 0x03FF;
#define BTB_SIZE 1024

struct entry_two_bits
{
    bool valid;
    bool prediction;
    UINT64 tag;
    UINT64 ReplaceCount;
    UINT8 counter;
} BTB[BTB_SIZE];

/* initialize the BTB */
VOID BTB_init()
{
  int i;

    for(i = 0; i < BTB_SIZE; i++)
    {
        BTB[i].valid = false;
        BTB[i].prediction = false;
        BTB[i].tag = 0;
        BTB[i].ReplaceCount = 0;
        BTB[i].counter = 0;
    }

  // HINT: in the case of the two-bit predictor, you would just need to update the prediction
  //  field to implement the two-bit automata

  // HINT: in the case of GAg, you will need to add a history register and history table
  //  something like:
  //     unsigned int BTB_History = 0;
  //     unsigned int BTB_HistoryLength = 8;
  //     unsigned char BTB_Table[256];
  // the table would be initialized to 2 for weak taken, and
  // you would access the table using:
  //
  //    prediction = BTB_Table[BTB_History & 0xFF];
  //    if (prediction > 1)
  //      prediction is taken
  //    else 
  //      prediction is not taken
  //
  // note: the code above only handles an 8-bit history
 
  // HINT: in the case of PAg, you will need to add a field to the BTB structure
  // to maintain the history for the branch of that entry
}

/* see if the given address is in the BTB */
bool BTB_lookup(ADDRINT ins_ptr)
{
    UINT64 index;

    index = mask & ins_ptr;

    if(BTB[index].valid)
        if(BTB[index].tag == ins_ptr)
            return true;

    return false;
}

/* return the prediction for the given address */
bool BTB_prediction(ADDRINT ins_ptr)
{
    UINT64 index;

    index = mask & ins_ptr;
    
    if(BTB[index].counter > 1){
      return true;
    } else {
      return false; 
    }
}

/* update the BTB entry with the last result */
VOID BTB_update(ADDRINT ins_ptr, bool taken)
{
    UINT64 index;
    UINT8 counter; 

    index = mask & ins_ptr;
    counter = BTB[index].counter;

    if (taken) { 
      counter ++;
      if (counter > 3) {
	counter = 3;
      }
    } else { 
      if (counter != 0) { 
	counter --;
      }
    }
    BTB[index].counter = counter;
}

/* insert a new branch in the table */
VOID BTB_insert(ADDRINT ins_ptr)
{
    UINT64 index;

    index = mask & ins_ptr;

    if(BTB[index].valid)
    {
        BTB[index].ReplaceCount++;
        CountReplaced++;
    }

    BTB[index].valid = true;
    BTB[index].prediction = true;  // Missed branches always enter as taken/true
    BTB[index].tag = ins_ptr;
    //BTB[index].counter = 2;
}

/* ===================================================================== */


/* ===================================================================== */

VOID WriteResults(bool limit_reached)
{
    int i;

    string output_file = KnobOutputFile.Value();
    if(KnobPid) output_file += "." + decstr(getpid());
    
    std::ofstream out(output_file.c_str());

    if(limit_reached)
        out << "Reason: limit reached\n";
    else
        out << "Reason: fini\n";
    out << "Count Seen: " << CountSeen << endl;
    out << "Count Taken: " << CountTaken << endl;
    out << "Count Correct: " << CountCorrect << endl;
    out << "Count Replaced: " << CountReplaced << endl;
    out << "Accuracy: " << (float) CountCorrect / CountSeen << endl;
    out << "Miss Rate " << (float) CountReplaced / CountSeen << endl;

    for(i = 0; i < BTB_SIZE; i++)
    {
        out << "BTB entry: " << i << ";" << BTB[i].valid << ";" << BTB[i].ReplaceCount << endl;
    }
    out.close();
}

/* ===================================================================== */

VOID br_predict(ADDRINT ins_ptr, INT32 taken)
{
    CountSeen++;
    if (taken)
        CountTaken++;

    if(BTB_lookup(ins_ptr)) 
    {
        if(BTB_prediction(ins_ptr) == taken) CountCorrect++;
        BTB_update(ins_ptr, taken);
    }
    else
    {
        if(!taken) CountCorrect++;
        else BTB_insert(ins_ptr);
    }

    if(CountSeen == KnobBranchLimit.Value())
    {
        WriteResults(true);
        exit(0);
    }
} 


//  IARG_INST_PTR   
// ADDRINT ins_ptr

/* ===================================================================== */

VOID Instruction(INS ins, void *v)
{

// The subcases of direct branch and indirect branch are
// broken into "call" or "not call".  Call is for a subroutine
// These are left as subcases in case the programmer wants
// to extend the statistics to see how sub cases of branches behave

    if( INS_IsRet(ins) )
    {
        INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR) br_predict, 
            IARG_INST_PTR, IARG_BRANCH_TAKEN,  IARG_END);
    }
    else if( INS_IsSyscall(ins) )
    {
        INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR) br_predict, 
            IARG_INST_PTR, IARG_BRANCH_TAKEN,  IARG_END);
    }
    else if (INS_IsDirectBranchOrCall(ins))
    {
        if( INS_IsCall(ins) ) {
            INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR) br_predict, 
                IARG_INST_PTR, IARG_BRANCH_TAKEN,  IARG_END);
        }
        else {
            INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR) br_predict, 
                IARG_INST_PTR, IARG_BRANCH_TAKEN,  IARG_END);
        }
    }
    else if( INS_IsIndirectBranchOrCall(ins) )
    {
        if( INS_IsCall(ins) ) {
            INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR) br_predict, 
                IARG_INST_PTR, IARG_BRANCH_TAKEN,  IARG_END);
    }
        else {
            INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR) br_predict, 
                IARG_INST_PTR, IARG_BRANCH_TAKEN,  IARG_END);
        }
    }

}

/* ===================================================================== */

#define OUT(n, a, b) out << n << " " << a << setw(16) << CountSeen. b  << " " << setw(16) << CountTaken. b << endl

VOID Fini(int n, void *v)
{
    WriteResults(false);
}


/* ===================================================================== */


/* ===================================================================== */

int main(int argc, char *argv[])
{
    
    if( PIN_Init(argc,argv) )
    {
        return Usage();
    }

    BTB_init(); // Initialize hardware structures
        
    INS_AddInstrumentFunction(Instruction, 0);
    PIN_AddFiniFunction(Fini, 0);

    // Never returns

    PIN_StartProgram();
    
    return 0;
}

/* ===================================================================== */
/* eof */
/* ===================================================================== */
