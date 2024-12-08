import traceback
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import numpy as np
from tensorflow import keras
from geopy.distance import geodesic
from decouple import config
import os

rootdir = os.path.join(os.getcwd(), "model")

recommend_destination_model = keras.models.load_model(os.path.join(rootdir, config('MODEL_1')))

df = pd.read_csv(os.path.join(rootdir, config('DATASET')))
df.dropna(axis=0, inplace=True)
df['latitude'] = df['latitude'].astype(float)
df['longitude'] = df['longitude'].astype(float)

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

            if not selected_place:
                return Response({"error": "selected_place is required"}, status=status.HTTP_400_BAD_REQUEST)

            selected_row = df[df["nama_destinasi"] == selected_place]
            if selected_row.empty:
                return Response({"error": f"Destinasi '{selected_place}' not found in the dataset."}, status=status.HTTP_404_NOT_FOUND)

            features = ['latitude', 'longitude', 'provinsi_encoded', 'kategori_encoded']
            X = df[features]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            df['predicted_rating'] = recommend_destination_model.predict(X_scaled)

            selected_coords = (selected_row.iloc[0]["latitude"], selected_row.iloc[0]["longitude"])
            df['distance_km'] = df.apply(
                lambda row: geodesic(selected_coords, (row["latitude"], row["longitude"])).kilometers,
                axis=1
            )

            recommendations = df.sort_values(by=["distance_km", "predicted_rating"], ascending=[True, False])
            recommendations = recommendations.head(num_recommendations)

            response_data = recommendations[[
                'nama_destinasi', 'letak_provinsi', 'kategori', 'average_rating', 'predicted_rating', 'distance_km'
            ]].to_dict(orient='records')

            return Response(response_data, status=status.HTTP_200_OK)

        except Exception as e:
            if config('DEBUG', cast=bool):
                traceback.print_exc()
            return Response({"error": f"An error occurred: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
topfive_similiar_destination_model = keras.models.load_model(os.path.join(rootdir, config('MODEL_2')))
        
class TopFiveSimiliarDestinationAPIView(APIView):
    def post(self, request):
        try:
            selected_place = request.data.get("selected_place")
            if not selected_place:
                return Response({"error": "Selected place is required"}, status=status.HTTP_400_BAD_REQUEST)

            selected_row = df[df['nama_destinasi'] == selected_place]
            if selected_row.empty:
                return Response({"error": f"Destination '{selected_place}' not found in the dataset."}, status=status.HTTP_404_NOT_FOUND)

            provinsi_encoded = selected_row['provinsi_encoded'].values[0]
            kategori_encoded = selected_row['kategori_encoded'].values[0]
            average_rating = selected_row['average_rating'].values[0]
            review_total = selected_row['review_total'].values[0]

            input_data = [
                np.array([provinsi_encoded]),
                np.array([kategori_encoded]),
                np.array([average_rating]),
                np.array([review_total])
            ]

            predicted_popularity = topfive_similiar_destination_model.predict(input_data)[0][0]

            similar_destinations = df[(df['provinsi_encoded'] == provinsi_encoded) & (df['kategori_encoded'] == kategori_encoded)]
            similar_destinations = similar_destinations.sort_values(by=['average_rating', 'review_total'], ascending=False)

            top_five = similar_destinations[['nama_destinasi', 'letak_provinsi', 'kategori', 'average_rating', 'review_total']].head(5)

            top_five_data = top_five.to_dict(orient="records")
            return Response({
                "selected_place": selected_place,
                "predicted_popularity": predicted_popularity,
                "top_five_similar_destinations": top_five_data
            }, status=status.HTTP_200_OK)

        except Exception as e:
            if config('DEBUG', cast=bool):
                traceback.print_exc()
            return Response({"error": f"An error occurred: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
class MostPopularDestination(APIView):
    def post(self, request):
        try:
            data = request.data
            province = data.get('province')
            category = data.get('category')

            if not province or not category:
                return Response(
                    {"error": "Both 'province' and 'category' are required."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            dataset_path = os.path.join(rootdir, config('DATASET'))
            model_path = os.path.join(rootdir, config('MODEL_3'))

            df = pd.read_csv(dataset_path)
            model = keras.models.load_model(model_path)

            filtered_df = df[(df['letak_provinsi'] == province) & (df['kategori'] == category)]

            if filtered_df.empty:
                return Response(
                    {"message": "No matching places found."},
                    status=status.HTTP_404_NOT_FOUND,
                )

            filtered_df['kategori_encoded'] = filtered_df['kategori'].astype('category').cat.codes
            filtered_df['provinsi_encoded'] = filtered_df['letak_provinsi'].astype('category').cat.codes

            input_features = filtered_df[['provinsi_encoded', 'kategori_encoded', 'average_rating', 'review_total']]

            filtered_df['predicted_popularity'] = model.predict(input_features)

            filtered_df = filtered_df.sort_values(by=['predicted_popularity'], ascending=False)

            recommended_places = filtered_df[['nama_destinasi', 'kategori', 'average_rating', 'predicted_popularity']].head(10)

            return Response(recommended_places.to_dict(orient='records'), status=status.HTTP_200_OK)

        except Exception as e:
            if config('DEBUG', cast=bool):
                traceback.print_exc()
            return Response({"error": f"An error occurred: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

def calculate_distance(lat1, lon1, lat2, lon2):
    coords_1 = (lat1, lon1)
    coords_2 = (lat2, lon2)
    return geodesic(coords_1, coords_2).kilometers

class ItineraryGeneratorAPIView(APIView):
    def post(self, request):
        try:
            theme = request.data.get('theme')
            region = request.data.get('region')
            num_destinations = request.data.get('num_destinations', 5)

            if not theme or not region:
                return Response({"error": "'theme' and 'region' are required."}, status=status.HTTP_400_BAD_REQUEST)

            dataset_path = os.path.join(rootdir, config('DATASET'))
            df = pd.read_csv(dataset_path)

            filtered_df = df[df['kategori'].str.contains(theme, case=False, na=False)]
            filtered_df = filtered_df[filtered_df['letak_provinsi'] == region]
            selected_places = filtered_df.sample(n=num_destinations)['nama_destinasi'].tolist()

            model_path = os.path.join(rootdir, config('MODEL_4'))
            model = keras.models.load_model(model_path)

            tokenizer = keras.preprocessing.text.Tokenizer()
            tokenizer.fit_on_texts(filtered_df['nama_destinasi'] + ', ' + filtered_df['kategori'] + ', ' + filtered_df['letak_provinsi'])

            prompt = f"Generate an itinerary for {theme} tourism in {region} including: {', '.join(selected_places)}"
            input_seq = tokenizer.texts_to_sequences([prompt])[0]
            for _ in range(20):
                input_array = keras.preprocessing.sequence.pad_sequences([input_seq], maxlen=15, padding='post')
                predicted_probs = model.predict(input_array, verbose=0)
                predicted_index = np.argmax(predicted_probs[0, -1])
                if not predicted_index:
                    break
                input_seq.append(predicted_index)

            itinerary_details = []
            visited_destinations = set()

            for i in range(len(selected_places) - 1):
                place1 = selected_places[i]
                place2 = selected_places[i + 1]

                if place1 in visited_destinations:
                    continue
                visited_destinations.add(place1)

                loc1 = df[df['nama_destinasi'] == place1].iloc[0]
                loc2 = df[df['nama_destinasi'] == place2].iloc[0]

                lat1, lon1 = loc1['latitude'], loc1['longitude']
                lat2, lon2 = loc2['latitude'], loc2['longitude']

                distance = calculate_distance(lat1, lon1, lat2, lon2)

                rating1 = loc1['average_rating']
                rating2 = loc2['average_rating']

                predicted_rating1 = (rating1 + rating2) / 2

                itinerary_details.append({
                    'nama_destinasi': place1,
                    'letak_provinsi': loc1['letak_provinsi'],
                    'kategori': loc1['kategori'],
                    'average_rating': rating1,
                    'predicted_rating': predicted_rating1,
                    'distance_km': 0 if i == 0 else distance
                })

            return Response(itinerary_details, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": f"An error occurred: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)