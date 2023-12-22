package com.example.lungsoundclassification;

import android.os.Bundle;
import android.view.View;
import android.widget.TextView;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.DefaultItemAnimator;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import java.util.ArrayList;
import java.util.List;

public class DiagnosisActivity extends AppCompatActivity {

    private List<DiagnosisModel> diagnosisList;
    private List<DiagnosisModel> viewableDiagnosisList;
    private DiagnosisAdapter adapter;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.diagnosis_view);

        RecyclerView recyclerView = findViewById(R.id.recyclerView);
        DefaultItemAnimator animator = new DefaultItemAnimator();
        animator.setAddDuration(200);
        animator.setRemoveDuration(200);

        recyclerView.setLayoutManager(new LinearLayoutManager(this));
        recyclerView.setItemAnimator(animator);

        List<DiagnosisModel> diagnosisList = getDiagnosisData(); // Replace with your data source
        adapter = new DiagnosisAdapter(diagnosisList);
        recyclerView.setAdapter(adapter);

        TextView expand_btn = findViewById(R.id.diagnosis_seemore);

        expand_btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // Handle button click
                if(expand_btn.getText().equals(getString(R.string.show_more))){
                    expandList();
                    expand_btn.setText(getString(R.string.show_less));

                } else {
                    minimizeList();
                    expand_btn.setText(getString(R.string.show_more));
                    adapter.notifyDataSetChanged();
                }
            }
        });

    }

    // Replace this method with your actual data source logic
    private List<DiagnosisModel> getDiagnosisData() {
        this.diagnosisList = new ArrayList<>();
        this.diagnosisList.add(new DiagnosisModel("Bronchiectasis", "5.01%"));
        this.diagnosisList.add(new DiagnosisModel("COPD", "12.12%"));
        this.diagnosisList.add(new DiagnosisModel("Pneumonia", "3.03%"));

        viewableDiagnosisList = new ArrayList<>();
        viewableDiagnosisList.add(diagnosisList.get(0));

        return viewableDiagnosisList;
    }

    private void expandList(){
        for(int i = 1; i < diagnosisList.size(); i++){
            viewableDiagnosisList.add(i, diagnosisList.get(i));
            adapter.notifyItemInserted(i);
        }

    }

    private void minimizeList(){
        for(int i = diagnosisList.size() - 1; i >= 1; i--){
            viewableDiagnosisList.remove(i);
            adapter.notifyItemRemoved(i);
        }
    }

}
