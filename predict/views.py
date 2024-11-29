from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import numpy as np
from tensorflow import keras
from geopy.distance import geodesic

# Load the TensorFlow model
nn_model = keras.models.load_model('predict/tensorflow_wisata_model_with_predictions.keras')

# Load the dataset
df = pd.read_csv("predict/dataset_fix.csv")
df.dropna(axis=0, inplace=True)
df['latitude'] = df['latitude'].astype(float)
df['longitude'] = df['longitude'].astype(float)

# Dynamically recreate label encoders
label_encoder_kategori = LabelEncoder()
df['kategori_encoded'] = label_encoder_kategori.fit_transform(df['kategori'])

label_encoder_provinsi = LabelEncoder()
df['provinsi_encoded'] = label_encoder_provinsi.fit_transform(df['letak_provinsi'])

class RecommendDestinationsAPIView(APIView):
    def post(self, request):
        try:
            data = request.data
            selected_place = data.get('selected_place')
            num_recommendations = data.get('num_recommendations', 5)

            # Validate input
            if not selected_place:
                return Response({"error": "selected_place is required"}, status=status.HTTP_400_BAD_REQUEST)

            # Find the selected destination in the dataset
            selected_row = df[df["nama_destinasi"] == selected_place]
            if selected_row.empty:
                return Response({"error": f"Destinasi '{selected_place}' not found in the dataset."}, status=status.HTTP_404_NOT_FOUND)

            # Preprocess features
            features = ['latitude', 'longitude', 'provinsi_encoded', 'kategori_encoded']
            X = df[features]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Predict ratings
            df['predicted_rating'] = nn_model.predict(X_scaled)

            # Find the recommendations
            selected_coords = (selected_row.iloc[0]["latitude"], selected_row.iloc[0]["longitude"])
            df['distance_km'] = df.apply(
                lambda row: geodesic(selected_coords, (row["latitude"], row["longitude"])).kilometers,
                axis=1
            )

            recommendations = df.sort_values(by=["distance_km", "predicted_rating"], ascending=[True, False])
            recommendations = recommendations.head(num_recommendations)

            # Prepare the response
            response_data = recommendations[[
                'nama_destinasi', 'letak_provinsi', 'kategori', 'average_rating', 'predicted_rating', 'distance_km'
            ]].to_dict(orient='records')

            return Response(response_data, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": f"An error occurred: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)